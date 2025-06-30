from typing import Tuple, Optional, Callable

import torch
import torch.fx as fx
from torch._functorch.aot_autograd import aot_function


class GraphCollector:

    def __init__(self):
        self.fwd_graph = None
        self.bwd_graph = None

    def get_forward_compiler(self):

        def _fw_compiler(fx_module, *args, **kwargs):
            self.fwd_graph = fx_module.graph
            return fx_module

        return _fw_compiler

    def get_backward_compiler(self):

        def _bw_compiler(fx_module, *args, **kwargs):
            self.bwd_graph = fx_module.graph
            return fx_module

        return _bw_compiler


class GraphOptimizer(fx.Transformer):

    def call_function(self, target, args, kwargs):
        if target.__name__ == "detach.default":
            # Remove `.detach()` operation.
            # We can safely remove it since the bwd graph has already been generated!
            i = args[0]
            return i
        return super().call_function(target, args, kwargs)


def _optimize_graph(graph: fx.Graph):
    return GraphOptimizer(fx.GraphModule({}, graph)).transform().graph


def generate_inference_graph(fn: Callable, example_inputs: tuple):
    collector = GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler()
    )

    for i in example_inputs:
        if isinstance(i, torch.Tensor):
            i.requires_grad = False

    # feed the fake inputs
    ys = f(*example_inputs)
    return _optimize_graph(collector.fwd_graph)


def generate_forward_and_backward_graph(
    fn: Callable,
    example_inputs: tuple,
    requires_grad: Optional[Tuple[bool]] = None
):
    collector = GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler()
    )

    # if requires_grad is specified, overwrite the requires_grad flag of the
    # tensors in example_inputs
    if requires_grad is not None:
        for i, r in zip(example_inputs, requires_grad):
            if isinstance(i, torch.Tensor):
                i.requires_grad = r

    # feed the fake inputs
    ys = f(*example_inputs)
    # choose a Tensor in ys as the starting point of .backward()
    o = None
    for y in ys:
        if isinstance(y, torch.Tensor):
            o = y
            break
    if o is None:
        raise ValueError(f"No Tensor found in the output of the function {fn}")
    # create a fake gradient
    g = torch.randn_like(o)
    # backward
    o.backward(g)

    collector.bwd_graph.lint()

    return (
        _optimize_graph(collector.fwd_graph),
        _optimize_graph(collector.bwd_graph),
    )
