from typing import Callable, Tuple
import tempfile
import os
from pathlib import Path

import torch
import torch.fx as fx
import triton
import triton.language as tl

import sys

sys.path.append("./")
from flashsnn.utils.dtype import type_str_dict


def _unwrap(arg: fx.Node) -> str:
    if isinstance(arg, fx.Node):
        return arg.name
    elif isinstance(arg, torch.dtype):
        return type_str_dict[arg]
    return str(arg)


FX_TO_TRITON = {
    'add': lambda node: f"{_unwrap(node.args[0])} + {_unwrap(node.args[1])}",
    'sub': lambda node: f"{_unwrap(node.args[0])} - {_unwrap(node.args[1])}",
    'mul': lambda node: f"{_unwrap(node.args[0])} * {_unwrap(node.args[1])}",
    'ge': lambda node: f"{_unwrap(node.args[0])} >= {_unwrap(node.args[1])}",
    'to': lambda node: f"{_unwrap(node.args[0])}.to({_unwrap(node.args[1])})",
    'sigmoid': lambda node: f"tl.sigmoid({_unwrap(node.args[0])})",
}

INDENTATIon = "    "


def generate_triton_code_str(
    fn: Callable,
    verbose: bool = False,
) -> Tuple[str, str]:
    traced = fx.symbolic_trace(fn)
    if verbose:
        print(traced.graph)

    inputs = []
    triton_code_lines = []
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.name)
        elif node.op == "call_function":
            op_name = node.target.__name__
            if op_name in FX_TO_TRITON:
                rhs = FX_TO_TRITON[op_name](node)
                triton_code_lines.append(f"{node.name} = {rhs}")
            else:
                raise NotImplementedError(
                    f"call_function {op_name} has not yet been implemented "
                    f"in FX_TO_TRITON mapping."
                )
        elif node.op == "call_method":
            op_name = node.target
            if op_name in FX_TO_TRITON:
                rhs = FX_TO_TRITON[op_name](node)
                triton_code_lines.append(f"{node.name} = {rhs}")
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
    signiture = ", ".join(inputs)
    signiture = f"@triton.jit\ndef {fn.__name__}({signiture}):"
    triton_code_lines = f"\n{INDENTATIon}".join(triton_code_lines)
    return (
        f"{prefix}\n\n{signiture}\n{INDENTATIon}{triton_code_lines}",
        fn.__name__,
    )


def compile_triton_code_str(
    triton_code: str,
    kernel_name: str,
    verbose: bool = False
) -> triton.JITFunction:
    # create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(triton_code)
        fpath = Path(f.name)
        if verbose:
            print(f"Triton code `{kernel_name}` written to {fpath}")

    try:
        name_space = {
            'triton': triton,
            'tl': tl,
        }
        with open(fpath, 'r') as f:
            code = compile(f.read(), fpath, 'exec')
            exec(code, name_space)

        if kernel_name in name_space:
            return name_space[kernel_name]
        else:
            raise ValueError(
                f"Function {kernel_name} not found in compiled namespace"
            )
    finally:
        # always remove the temporary file!
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
    kernel_str, kernel_name = generate_triton_code_str(fn, verbose)
    if verbose:
        print("=" * 100)
        print("Generated Triton code:\n```")
        print(kernel_str)
        print("```")
        print("=" * 100)
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe
