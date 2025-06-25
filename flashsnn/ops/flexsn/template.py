from typing import List

import triton

from flashsnn.ops.torch2triton import compile_triton_code_str

INDENTATION = " " * 4

inference_template = """{core_str}

@triton.jit
def flexsn_inference_kernel_{hash}(
    x_seq_ptr,  # [T, NCL]
    s_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v = tl.zeros([BLOCK_NCL], dtype=dtype)

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        s, v = {core_name}(x, v)

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))
"""


def get_flexsn_inference_kernel(
    core_str: str, core_name: str, verbose: bool = False
) -> triton.JITFunction:
    hash = core_name[-8:]
    kernel_str = inference_template.format(
        core_str=core_str, core_name=core_name, hash=hash
    )
    kernel_name = f"flexsn_inference_kernel_{hash}"
    if verbose:
        print("Generated flexsn inference kernel:")
        print("```")
        print(kernel_str)
        print("```\n")
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe


forward_template = """{core_str}

@triton.jit
def flexsn_forward_kernel_{hash}(
    x_seq_ptr,  # [T, NCL]
    s_seq_ptr,
    {extra_signature},
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v = tl.zeros([BLOCK_NCL], dtype=dtype)

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        s, v{extra_core_return} = {core_name}(x, v)

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))
        {extra_store}
"""

store_template = """
        {name}_ptrs = tl.make_block_ptr(
            {name}_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store({name}_ptrs, {name}, boundary_check=(1,))
"""


def get_flexsn_forward_kernel(
    core_str: str,
    core_name: str,
    bi2fo: List[int],  # flashsnn.ops.torch2triton.auto_backward.get_bi2fo()
    verbose: bool = False,
) -> triton.JITFunction:
    hash = core_name[-8:]

    # Collect bwd_core's required inputs. s_seq should bot be included here, as
    # it has been included in the forward kernel. We use bi2fo[j] > 0 to filter
    # out s_seq (bi2fo==0) and unmapped values. Notice that extra_signature
    # will follow bi2fo's order (a.k.a. bwd_core's input order).
    extra_signature = f",\n{INDENTATION}".join([
        (f"v_seq_ptr" if i == 1 else f"res{i}_seq_ptr") for i in bi2fo if i > 0
    ])

    # This is bi2fo's value domain! res0 is s, res1 is v; these two special
    # values are written to the template and thus are not included here.
    extra_core_return = "".join([f", res{i}" for i in range(2, max(bi2fo) + 1)])

    # Store the required results specified by bi2fo. s is always stored and is
    # written to the template, so it is not included here.
    extra_store = "".join([
        store_template.format(name=f"v" if i == 1 else f"res{i}")
        for i in bi2fo
        if i > 0
    ])

    kernel_str = forward_template.format(
        core_str=core_str,
        core_name=core_name,
        hash=hash,
        extra_signature=extra_signature,
        extra_core_return=extra_core_return,
        extra_store=extra_store
    )
    kernel_name = f"flexsn_forward_kernel_{hash}"
    if verbose:
        print("Generating flexsn forward kernel:")
        print("```")
        print(kernel_str)
        print("```")
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe


backward_template = """{core_str}

@triton.jit
def flexsn_backward_kernel_{hash}(
    grad_s_seq_ptr,
    {extra_signature},
    grad_x_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_s_ptrs = tl.make_block_ptr(
            grad_s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        grad_s = tl.load(
            grad_s_ptrs, boundary_check=(1,), padding_option="zero"
        )
        {extra_load}
        grad_x, grad_v = {core_name}(grad_s, grad_v{extra_core_input})

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_x, boundary_check=(1,))
"""

load_template = """
        {name}_ptrs = tl.make_block_ptr(
            {name}_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        {name} = tl.load(
            {name}_ptrs, boundary_check=(1,), padding_option="zero"
        )
"""


def get_flexsn_backward_kernel(
    core_str: str,
    core_name: str,
    bi2fo: str,  # flashsnn.ops.torch2triton.auto_backward.get_bi2fo()
    verbose: bool = False,
) -> triton.JITFunction:
    hash = core_name[-8:]

    # Set bwd kernel's signature according to bwd_core's required inputs. Notice
    # that the order of these arguments follows bi2fo's order (a.k.a.
    # bwd_core's input order).
    extra_signature = f",\n{INDENTATION}".join([
        f"res{i}_seq_ptr" for i in bi2fo if i >= 0
    ])

    # Load required intermediate results.
    extra_load = "".join([
        load_template.format(name=f"res{i}") for i in bi2fo if i >= 0
    ])

    # Set bwd_core's input according to bi2fo.
    extra_core_input = "".join([f", res{i}" for i in bi2fo if i >= 0])

    kernel_str = backward_template.format(
        core_str=core_str,
        core_name=core_name,
        hash=hash,
        extra_signature=extra_signature,
        extra_load=extra_load,
        extra_core_input=extra_core_input,
    )
    kernel_name = f"flexsn_backward_kernel_{hash}"
    if verbose:
        print("Generating flexsn forward kernel:")
        print("```")
        print(kernel_str)
        print("```")
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)
    return kernel_exe
