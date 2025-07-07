import triton

from flashsnn.torch2triton import compile_triton_code_str

INDENTATION = " " * 4

store_template = """
        {name}_ptrs = tl.make_block_ptr(
            {name}_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        convert_and_store({name}_ptrs, {name}, boundary_check=(1,))
        # tl.store({name}_ptrs, {name}, boundary_check=(1,))
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

kernel_template = """from flashsnn.ops.snippets import convert_and_store
{core_str}


@triton.autotune(
    configs=[
        triton.Config({{"BLOCK_NCL": f * w * 32}}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype"],
    restore_value=[{autotune_restore}],
)
@triton.jit
def flexsn_{kernel_type}_kernel_{hash}(
    {input_signature},
    {return_signature},
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    {state_initialization}

    for t in tl.static_range({loop_range}):
        {loads}

        {lhs} = {core_name}({core_args})

        {stores}
"""


def get_flexsn_inference_kernel(
    core_str: str,
    core_name: str,
    info: dict,
    verbose: bool = False
) -> triton.JITFunction:
    hash = core_name[-8:]
    num_inputs = info["num_inputs"]
    num_states = info["num_states"]
    num_outputs = info["num_outputs"]

    input_signature = f",\n{INDENTATION}".join([
        f"x{i}_seq_ptr" for i in range(num_inputs)
    ])
    return_signature = f",\n{INDENTATION}".join([
        f"s{i}_seq_ptr" for i in range(num_outputs)
    ])
    autotune_restore = f", ".join([
        f'"s{i}_seq_ptr"' for i in range(num_states)
    ])
    state_initialization = f"\n{INDENTATION}".join([
        f"v{i} = tl.zeros([1, BLOCK_NCL], dtype=dtype)"
        for i in range(num_states)
    ])
    loads = "".join([
        load_template.format(name=f"x{i}") for i in range(num_inputs)
    ])
    stores = "".join([
        store_template.format(name=f"s{i}") for i in range(num_outputs)
    ])
    lhs = ", ".join([f"s{i}" for i in range(num_outputs)])
    lhs += ", "
    lhs += ", ".join([f"v{i}" for i in range(num_states)])
    core_args = ", ".join([f"x{i}" for i in range(num_inputs)])
    core_args += ", "
    core_args += ", ".join([f"v{i}" for i in range(num_states)])

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_restore=autotune_restore,
        kernel_type="inference",
        hash=hash,
        input_signature=input_signature,
        return_signature=return_signature,
        state_initialization=state_initialization,
        loop_range="0, T, 1",
        loads=loads,
        lhs=lhs,
        core_name=core_name,
        core_args=core_args,
        stores=stores,
    ).strip()
    kernel_name = f"flexsn_inference_kernel_{hash}"
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generated flexsn inference kernel:")
        print("```")
        print(kernel_str)
        print("```\n")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    return kernel_exe


def get_flexsn_forward_kernel(
    core_str: str,
    core_name: str,
    info: dict,
    verbose: bool = False,
) -> triton.JITFunction:
    hash = core_name[-8:]
    num_inputs = info["num_inputs"]
    num_states = info["num_states"]
    fwd_kernel_returns = info["fwd_kernel_returns"]
    fwd_core_return_symbols = info["fwd_core_return_symbols"]

    input_signature = f",\n{INDENTATION}".join([
        f"x{i}_seq_ptr" for i in range(num_inputs)
    ])
    return_signature = f",\n{INDENTATION}".join([
        f"{r}_seq_ptr" for r in fwd_kernel_returns
    ])
    autotune_restore = ", ".join([f'"{s}_seq_ptr"' for s in fwd_kernel_returns])
    state_initialization = f"\n{INDENTATION}".join([
        f"v{i} = tl.zeros([1, BLOCK_NCL], dtype=dtype)"
        for i in range(num_states)
    ])
    loads = "".join([
        load_template.format(name=f"x{i}") for i in range(num_inputs)
    ])
    lhs = ", ".join([r for r in fwd_core_return_symbols])
    core_args = ", ".join([f"x{i}" for i in range(num_inputs)])
    core_args += ", "
    core_args += ", ".join([f"v{i}" for i in range(num_states)])
    stores = "".join([
        store_template.format(name=r) for r in fwd_kernel_returns
    ])

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_restore=autotune_restore,
        kernel_type="forward",
        hash=hash,
        input_signature=input_signature,
        return_signature=return_signature,
        state_initialization=state_initialization,
        loop_range="0, T, 1",
        loads=loads,
        lhs=lhs,
        core_name=core_name,
        core_args=core_args,
        stores=stores,
    ).strip()
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


def get_flexsn_backward_kernel(
    core_str: str,
    core_name: str,
    info: dict,
    verbose: bool = False,
) -> triton.JITFunction:
    hash = core_name[-8:]
    num_outputs = info["num_outputs"]
    num_inputs = info["num_inputs"]
    num_states = info["num_states"]
    n = len(info["extra_return_mapping"])  # number of required results

    input_signature = f",\n{INDENTATION}".join([
        f"grad_s{i}_seq_ptr" for i in range(num_outputs)
    ])
    input_signature += f",\n{INDENTATION}"
    input_signature += f",\n{INDENTATION}".join([
        f"res{i}_seq_ptr" for i in range(n)
    ])

    return_signature = f",\n{INDENTATION}".join([
        f"grad_x{i}_seq_ptr" for i in range(num_inputs)
    ])

    autotune_restore = f", ".join([
        f'"grad_x{i}_seq_ptr"' for i in range(num_inputs)
    ])

    state_initialization = f"\n{INDENTATION}".join([
        f"grad_v{i} = tl.zeros([1, BLOCK_NCL], dtype=dtype)"
        for i in range(num_states)
    ])

    loads = "".join([
        load_template.format(name=f"grad_s{i}") for i in range(num_outputs)
    ])
    loads += "".join([load_template.format(name=f"res{i}") for i in range(n)])

    stores = "".join([
        store_template.format(name=f"grad_x{i}") for i in range(num_inputs)
    ])

    lhs = ", ".join([f"grad_x{i}" for i in range(num_inputs)])
    lhs += ", "
    lhs += ", ".join([f"grad_v{i}" for i in range(num_states)])

    core_args = ", ".join([f"res{i}" for i in range(n)])
    core_args += ", "
    core_args += ", ".join([f"grad_s{i}" for i in range(num_outputs)])
    core_args += ", "
    core_args += ", ".join([f"grad_v{i}" for i in range(num_states)])

    kernel_str = kernel_template.format(
        core_str=core_str,
        autotune_restore=autotune_restore,
        kernel_type="backward",
        hash=hash,
        input_signature=input_signature,
        return_signature=return_signature,
        state_initialization=state_initialization,
        loop_range="T-1, -1, -1",
        loads=loads,
        lhs=lhs,
        core_name=core_name,
        core_args=core_args,
        stores=stores,
    ).strip()
    kernel_name = f"flexsn_backward_kernel_{hash}"
    kernel_exe = compile_triton_code_str(kernel_str, kernel_name, verbose)

    if verbose:
        print("=" * 40, core_name, "=" * 40)
        print("Generated flexsn backward kernel:")
        print("```")
        print(kernel_str)
        print("```\n")
        print(info)
        print("=" * 40, "=" * len(core_name), "=" * 40)

    return kernel_exe
