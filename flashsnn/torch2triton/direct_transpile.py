from typing import Callable, Tuple, Union, Optional
import tempfile
from pathlib import Path
import hashlib

import torch
import torch.fx as fx
import triton
import triton.language as tl

from flashsnn.utils.dtype import type_str_dict
from flashsnn.utils.cleanup import ensure_cleanup_tmp_python_files


def _generate_hash(s: str, w: int = 8) -> str:
    hasher = hashlib.sha256(s.encode("utf-8"))
    return hasher.hexdigest()[:w]


def _uw(arg) -> str:  # unwrap
    if isinstance(arg, fx.Node):
        return arg.name
    elif isinstance(arg, torch.dtype):
        return type_str_dict[arg]
    return str(arg)


PI = 3.14159265358979

# code generation rules
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
    "detach": # do not need to define "p_detach"; skip node generation instead
        lambda args: f"{_uw(args[0])}",
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
            f"({_uw(args[0])} / "
            f"tl.fma({PI}*{_uw(args[1])}, {PI}*{_uw(args[1])}, 1.))"
            f".to({_uw(args[1])}.dtype)"
        ),
}

INDENTATION = " " * 4  # four spaces


def generate_triton_code_str(
    graph: Union[fx.Graph, Callable],
    fn_name: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[str, str]:
    """Given a fx.Graph, generate its corresponding Triton code string.

    Args:
        graph (fx.Graph or Callable): if a function is given, convert it to 
            fx.Graph first.
        fn_name (str): name of the original PyTorch function. If None and `graph`
            is a function, `fn_name` will be set to the function name.
        verbose (bool, optional): Defaults to False.

    Returns:
        Tuple[str, str]: the generated Triton code string and the name of the 
            Triton function.
    """
    if not isinstance(graph, fx.Graph):
        fn_name = graph.__name__
        graph = fx.symbolic_trace(graph).graph
    if verbose:
        print(graph)

    inputs = []
    triton_code_lines = []
    for node in graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.name)
        elif node.op in ["call_function", "call_method"]:
            # For registered custom ops, node.target or node.target.__name__
            # yields "op_name.default" or something like that. Erase the postfix
            # using `split(".")[0]`.
            op_name = (
                node.target.__name__
                if node.op == "call_function" else node.target
            ).split(".")[0]
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

    triton_code_lines = f"{INDENTATION}" + f"\n{INDENTATION}".join(
        triton_code_lines
    )
    fn_name = f"{fn_name}_{_generate_hash(triton_code_lines)}"
    signature = ", ".join(inputs)
    signature = f"@triton.jit\ndef {fn_name}({signature}):"
    prefix = "import triton\nimport triton.language as tl"
    return f"{prefix}\n\n{signature}\n{triton_code_lines}", fn_name


@ensure_cleanup_tmp_python_files
def compile_triton_code_str(
    triton_code: str,
    kernel_name: str,
    verbose: bool = False,
    name_space: dict = {},
) -> triton.JITFunction:
    # create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(triton_code)
        fpath = Path(f.name)
        if verbose:
            print(f"Triton code `{kernel_name}` written to {fpath}")

    try:
        name_space.update({
            "triton": triton,
            "tl": tl,
            "__name__": "flashsnn.codegen.triton",  # TODO: any better choice?
        })
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
    kernel_exe = compile_triton_code_str(
        kernel_str, kernel_name, verbose=verbose
    )
    return kernel_exe
