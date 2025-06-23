import triton
import triton.language as tl

INDENTATION = " " * 4

inference_template = """
@triton.jit
def _spiking_neuron_inference_kernel(
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

        s, v = {core}(x, v)

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


def get_spiking_neuron_inference_kernel(core: triton.JITFunction):
    return inference_template.format(core=core)


forward_template = """
@triton.jit
def _spiking_neuron_inference_kernel(
    x_seq_ptr,  # [T, NCL]
    s_seq_ptr,
    {intermediate_result_signature},
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

        s, v{intermediate_result_core} = {core}(x, v)

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))
        {intermediate_result_store}
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


def get_spiking_neuron_forward_kernel(
    core: triton.JITFunction, intermediate_result_names
):
    intermediate_result_signature = f",\n{INDENTATION}".join([
        f"{name}_seq_ptr" for name in intermediate_result_names
    ])
    intermediate_result_core = "".join([
        f", {name}" for name in intermediate_result_names
    ])
    intermediate_result_store = "".join([
        store_template.format(name=name) for name in intermediate_result_names
    ])

    return forward_template.format(
        core=core,
        intermediate_result_signature=intermediate_result_signature,
        intermediate_result_core=intermediate_result_core,
        intermediate_result_store=intermediate_result_store
    )


backward_template = """
@triton.jit
def _spiking_neuron_backward_kernel(
    grad_s_seq_ptr,
    {intermediate_result_signature},
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
        {intermediate_result_load}
        grad_x, grad_v = {core}(grad_s, grad_v{intermediate_result_core})

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


def get_spiking_neuron_backward_kernel(
    core: triton.JITFunction, intermediate_result_names
):
    intermediate_result_signature = f",\n{INDENTATION}".join([
        f"{name}_seq_ptr" for name in intermediate_result_names
    ])
    intermediate_result_load = "".join([
        load_template.format(name=name) for name in intermediate_result_names
    ])
    intermediate_result_core = "".join([
        f", {name}" for name in intermediate_result_names
    ])

    return backward_template.format(
        core=core,
        intermediate_result_signature=intermediate_result_signature,
        intermediate_result_load=intermediate_result_load,
        intermediate_result_core=intermediate_result_core,
    )
