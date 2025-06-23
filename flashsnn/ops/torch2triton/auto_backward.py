from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.fx as fx
import triton

from flashsnn.ops.torch2triton.transpile import generate_triton_code_str
from flashsnn.ops.torch2triton.transpile import compile_triton_code_str

# key: forward operator name
# value: Callable node -> tuple
#   - node: the forward node *args -> z
#   - returns a sequence of tuples (grad_op_name, *args)
#     - grad_op_name: the name of the gradient operation
#     - *args: the saved results required by the gradient operation,
#       expressed as the keys of the dict `saved_results`.
BACKWARD_RULES = {
    'add':
        lambda node: (
            ("p_add_1", []),  # dx = dz
            ("p_add_2", []),  # dy = dz
        ),
    'sub':
        lambda node: (
            ("p_sub_1", []),  # dx = dz
            ("p_sub_2", [])  # dy = -dz
        ),
    'mul':
        lambda node: (
            ("p_mul_1", [node.args[1].name]),  # dx = dz * y
            ("p_mul_2", [node.args[0].name])  # dy = dz * x
        ),
    'sigmoid':
        lambda node: (
            ("p_sigmoid", [node.name]),  # dx = dz * z * (1 - z)
        ),
    'to':
        lambda node: (
            ("p_to", [node.args[0].name + "_dtype"]),  # dx = dz.to(x.dtype)
            ("0", []),
        ),
    'spike_fn':
        lambda node: (
            ("p_spike_fn", [node.args[0].name]),  # dx = dz * spike_fn(x)
        ),
}


def generate_backward_fx_graph(
    forward_graph: fx.Graph, requires_grad: Tuple[bool]
) -> fx.Graph:
    backward_graph = fx.Graph()

    # scan the forward graph, and identify the inputs to the backward graph
    # 1. grad_output(s)
    # 2. saved results (forward inputs and intermediate results)

    # grad_output(s) should be placed at the beginning!
    grad_nodes = {}  # forward node name -> gradient fx.Node in backward graph
    for node in forward_graph.nodes:
        if node.op == "output":  # create placeholders for grad_output(s)
            output_args = node.args[0]
            if isinstance(output_args, fx.Node):
                output_args = (output_args,)
            for output_arg in output_args:  # fx.Node
                grad_node = backward_graph.placeholder(
                    f"grad_{output_arg.name}_", type_expr=output_arg.type
                )
                grad_nodes[output_arg.name] = grad_node

    saved_results = {}  # forward node name -> saved fx.Node in backward graph
    for node in forward_graph.nodes:
        if node.op in ["placeholder", "call_function", "call_method"]:
            saved_results[node.name] = backward_graph.placeholder(
                node.name,
                type_expr=node.type,
            )
        if node.op == "call_method" and node.target == "to":
            # save the original dtype
            x_name = node.args[0].name
            saved_results[x_name + "_dtype"] = backward_graph.placeholder(
                x_name + "_dtype", type_expr=torch.dtype
            )

    # deal with the computations, reversely
    for node in reversed(forward_graph.nodes):
        if not node.op in ["call_function", "call_method"]:
            continue

        # op_name(*args) -> z
        op_name = (
            node.target.__name__ if node.op == "call_function" else node.target
        )
        if op_name not in BACKWARD_RULES:
            raise NotImplementedError(
                f"Backward rule for {op_name} has not yet been not implemented."
            )
        dz = grad_nodes.get(node.name, None)
        if dz is None:
            raise ValueError(
                f"Gradient for node {node.name} is not found "
                f"in the backward graph."
            )

        grad_ops_and_args = BACKWARD_RULES[op_name](node)

        for i, arg in enumerate(node.args):
            if not isinstance(arg, fx.Node):
                continue
            grad_op, grad_args = grad_ops_and_args[i]
            if grad_op == "0":
                continue
            grad_args_in_backward_graph = []
            for grad_arg in grad_args:  # forward graph nodes -> backward graph nodes
                grad_args_in_backward_graph.append(saved_results[grad_arg])

            grad_node = backward_graph.create_node(
                op="call_method",
                target=grad_op,
                # grad_output, plus required saved results
                args=(dz, *grad_args_in_backward_graph),
                name=f"grad_{arg.name}_"
            )
            # check if grad_{arg.name} already exists
            if arg.name in grad_nodes:
                # if True, accumulate the gradient!
                existing_grad_node = grad_nodes[arg.name]
                acc_grad_node = backward_graph.create_node(
                    op="call_method",
                    target="add",
                    args=(existing_grad_node, grad_node),
                    name=f"grad_{arg.name}_"
                )
                grad_nodes[arg.name] = acc_grad_node
            else:
                # if False, add the gradient node
                grad_nodes[arg.name] = grad_node

    fwd_placeholders = forward_graph.find_nodes(op="placeholder")
    output_grads = [
        grad_nodes.get(node.name)
        for i, node in enumerate(fwd_placeholders)
        if (node.name in grad_nodes) and requires_grad[i]
    ]
    backward_graph.output(tuple(output_grads), type_expr=None)

    # Now, the backward graph is ready, but redundant nodes may exist.
    # 1. Eliminate dead code (intermediate nodes whose num_users is 0)
    backward_graph.eliminate_dead_code()
    # 2. Eliminate redundant placeholders!
    for p in backward_graph.find_nodes(op="placeholder"):
        if len(p.users) == 0:
            backward_graph.erase_node(p)

    return backward_graph


def generate_backward_triton_code(
    fn: Callable,
    requires_grad: Tuple[bool],
    verbose: bool = False
) -> triton.JITFunction:
    """Given a PyTorch function, generate its BP's Triton JIT function.

    torch2triton module is still in development. Only a limited set of PyTorch
    operations (mainly element-wise operations) are supported currently.

    Args:
        fn (Callable): a PyTorch function.
        requires_grad (Tuple[bool]): a tuple of bools indicating whether the
            inputs of the function require gradients.
        verbose (bool, optional): If True, print the generated Triton code. 
            Defaults to False.

    Returns:
        triton.JITFunction
    """
    traced = fx.symbolic_trace(fn)
    backward_graph = generate_backward_fx_graph(traced.graph, requires_grad)

    kernel_str, kernel_name = generate_triton_code_str(
        backward_graph, fn.__name__, verbose
    )
    if verbose:
        print("=" * 100)
        print("Generated Triton code:\n```")
        print(kernel_str)
        print("```")
        print("=" * 100)
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe
