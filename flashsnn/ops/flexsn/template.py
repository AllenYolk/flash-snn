import triton

from flashsnn.torch2triton import compile_triton_code_str

INDENTATION = " " * 4

inference_template = """{core_str}

@triton.autotune(
    configs=[
        triton.Config({{}}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=["T", "BLOCK_NCL", "dtype"],
)
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

@triton.autotune(
    configs=[
        triton.Config({{}}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=["T", "BLOCK_NCL", "dtype"],
)
@triton.jit
def flexsn_forward_kernel_{hash}(
    x_seq_ptr,  # [T, NCL]
    {return_signature},
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

        {core_return} = {core_name}(x, v)

        {stores}
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
    info: dict,
    verbose: bool = False,
) -> triton.JITFunction:
    hash = core_name[-8:]
    fwd_kernel_returns = info["fwd_kernel_returns"]
    fwd_core_return_symbols = info["fwd_core_return_symbols"]

    return_signature = f",\n{INDENTATION}".join([
        f"{r}_seq_ptr" for r in fwd_kernel_returns
    ])
    core_return = ", ".join([r for r in fwd_core_return_symbols])
    stores = "".join([
        store_template.format(name=r) for r in fwd_kernel_returns
    ])

    kernel_str = forward_template.format(
        core_str=core_str,
        core_name=core_name,
        hash=hash,
        return_signature=return_signature,
        core_return=core_return,
        stores=stores,
    )
    kernel_name = f"flexsn_forward_kernel_{hash}"
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generating flexsn forward kernel:")
        print("```")
        print(kernel_str)
        print("```")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    return kernel_exe


backward_template = """{core_str}

@triton.autotune(
    configs=[
        triton.Config({{}}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=["T", "BLOCK_NCL", "dtype"],
)
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
        grad_x, grad_v = {core_name}({extra_core_input}grad_s, grad_v)

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
    info: dict,
    verbose: bool = False,
) -> triton.JITFunction:
    hash = core_name[-8:]
    n = info["N_fwd_core_returns"] - 2  # s, v not included

    # fwd_core_returns[2:] are all unique!!!
    extra_signature = f",\n{INDENTATION}".join([
        f"res{i}_seq_ptr" for i in range(n)
    ])

    # Load required intermediate results.
    extra_load = "".join([
        load_template.format(name=f"res{i}") for i in range(n)
    ])

    # Set bwd_core's input according to bi2fo.
    extra_core_input = "".join([f"res{i}, " for i in range(n)])

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
