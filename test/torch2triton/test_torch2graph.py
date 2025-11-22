import sys

sys.path.append("./")

import torch
import torch.fx as fx
from spikingjelly.activation_based import surrogate

from flashsnn import torch2triton

spike_fn = surrogate.ATan(alpha=2.)


def lif_core_generator(beta):

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        h = v*beta + x
        s = spike_fn(h - 1.)
        h = torch.where((s > 0.5) | ~(h < s), h - 1, h)
        return s, h

    return lif_core


def lif_core2(x: torch.Tensor, v: torch.Tensor, beta: torch.Tensor):
    h = v*beta + x
    s = spike_fn(h - 1.)
    v = h * (1. - s.detach())
    return s, v


if __name__ == "__main__":
    lif_core = lif_core_generator(beta=0.5)

    shape = (3, 4)
    h = torch.randn(shape, requires_grad=True)
    x = torch.randn(shape)
    g0 = torch2triton.generate_inference_graph(lif_core, example_inputs=(h, x))
    g1, g2 = torch2triton.generate_forward_and_backward_graph(
        lif_core, example_inputs=(h, x), requires_grad=(True, True)
    )
    print(g0)
    print(g1)
    print(g2)

    for node in g1.nodes:
        if node.op == "call_function":
            print(node.op, node.target.__name__)

    print(torch2triton.generate_triton_code_str(g0, "lif_core")[0])
