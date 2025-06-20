from typing import Callable, Tuple
import tempfile
import os
from pathlib import Path

import torch
import torch.fx as fx
import triton
import triton.language as tl

from flashsnn.utils.dtype import type_str_dict


def _uw(arg) -> str:  # unwrap
    if isinstance(arg, fx.Node):
        return arg.name
    elif isinstance(arg, torch.dtype):
        return type_str_dict[arg]
    return str(arg)


PI = 3.14159265358979

FX_TO_TRITON = {
    # forward
    "add":
        lambda args: f"{_uw(args[0])} + {_uw(args[1])}",
    "sub":
        lambda args: f"{_uw(args[0])} - {_uw(args[1])}",
    "mul":
        lambda args: f"{_uw(args[0])} * {_uw(args[1])}",
    "ge":
        lambda args: f"{_uw(args[0])} >= {_uw(args[1])}",
    "to":
        lambda args: f"{_uw(args[0])}.to({_uw(args[1])})",
    "sigmoid":
        lambda args: f"tl.sigmoid({_uw(args[0])})",
    "spike_fn":
        lambda args: f"({_uw(args[0])} >= 0.).to({_uw(args[0])}.dtype)",
    # backward
    "p_add_1":
        lambda args: f"{_uw(args[0])}",
    "p_add_2":
        lambda args: f"{_uw(args[0])}",
    "p_sub_1":
        lambda args: f"{_uw(args[0])}",
    "p_sub_2":
        lambda args: f"-{_uw(args[0])}",
    "p_mul_1":
        lambda args: f"{_uw(args[0])} * {_uw(args[1])}",
    "p_mul_2":
        lambda args: f"{_uw(args[0])} * {_uw(args[1])}",
    "p_sigmoid":
        lambda args:
        (f"{_uw(args[0])} * {_uw(args[1])} * (1 - {_uw(args[1])})"),
    "p_to":
        lambda args: f"{_uw(args[0])}.to({_uw(args[1])})",
    "p_spike_fn":
        lambda args: (
            f"(1. / tl.fma({PI}*{_uw(args[0])}, {PI}*{_uw(args[0])}, 1.))"
            f".to({_uw(args[0])}.dtype)"
        ),
}

INDENTATION = "    "


def generate_triton_code_str(
    graph: fx.Graph,
    fn_name: str,
    verbose: bool = False,
) -> Tuple[str, str]:
    if verbose:
        print(graph)

    inputs = []
    triton_code_lines = []
    for node in graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.name)
        elif node.op in ["call_function", "call_method"]:
            op_name = (
                node.target.__name__
                if node.op == "call_function" else node.target
            )
            if op_name in FX_TO_TRITON:
                rhs = FX_TO_TRITON[op_name](node.args)
                triton_code_lines.append(f"{node.name} = {rhs}")
            else:
                raise NotImplementedError(
                    f"{node.op} {op_name} has not yet been implemented "
                    f"in FX_TO_TRITON mapping."
                )
        elif node.op == "output":
            if isinstance(node.args[0], fx.Node):
                # only one return value
                things = node.args[0].name
            else:
                things = ", ".join(arg.name for arg in node.args[0])
            triton_code_lines.append(f"return {things}")
        else:
            raise NotImplementedError(
                f"Operation {node.op} has not yet been implemented."
            )

    prefix = "import triton\nimport triton.language as tl"
    signature = ", ".join(inputs)
    signature = f"@triton.jit\ndef {fn_name}({signature}):"
    triton_code_lines = f"\n{INDENTATION}".join(triton_code_lines)
    return (
        f"{prefix}\n\n{signature}\n{INDENTATION}{triton_code_lines}",
        fn_name,
    )


def compile_triton_code_str(
    triton_code: str,
    kernel_name: str,
    verbose: bool = False
) -> triton.JITFunction:
    # create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(triton_code)
        fpath = Path(f.name)
        if verbose:
            print(f"Triton code `{kernel_name}` written to {fpath}")

    try:
        name_space = {
            "triton": triton,
            "tl": tl,
        }
        with open(fpath, "r") as f:
            code = compile(f.read(), fpath, "exec")
            exec(code, name_space)

        if kernel_name in name_space:
            return name_space[kernel_name]
        else:
            raise ValueError(
                f"Function {kernel_name} not found in compiled namespace"
            )
    finally:
        # if the temporary file is removed,
        # triton will raise "source code not found" error
        # os.remove(fpath)
        pass


def transpile_triton_code(
    fn: Callable, verbose: bool = False
) -> triton.JITFunction:
    """Given a PyTorch function, generate its corresponding Triton JIT function.

    torch2triton module is still in development. Only a limited set of PyTorch
    operations (mainly element-wise operations) are supported currently.

    Args:
        fn (Callable): a PyTorch function.
        verbose (bool, optional): If True, print the generated Triton code. 
            Defaults to False.

    Returns:
        triton.JITFunction
    """
    traced = fx.symbolic_trace(fn)
    kernel_str, kernel_name = generate_triton_code_str(
        traced.graph, fn.__name__, verbose
    )
    if verbose:
        print("=" * 100)
        print("Generated Triton code:\n```")
        print(kernel_str)
        print("```")
        print("=" * 100)
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe
