import pytest
import sys

sys.path.append("./")

import torch
import triton
import triton.language as tl

from flashsnn.ops import torch2triton
from flashsnn.ops import surrogate_kernels, lif
from flashsnn.utils import assert_close, type_dict

SHAPE_LIST = [(4, 3, 3, 224, 224), (17, 5, 700)]
DTYPE_LIST = [torch.float32, torch.float16]
BETA_LIST = [0.5, 0.9, 0.1]


def sigmoid_surrogate_torch(
    h: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    alpha = 4.
    sgax = torch.sigmoid(
        alpha * h.to(torch.float32)
    )  # triton's exp() supports only fp32 and fp64. Manually convert it!
    sgax = sgax * (1.-sgax) * alpha
    return sgax.to(dtype)


@triton.jit
def sg_high_level_kernel(
    h_ptr, sg_ptr, N, BLOCK_SIZE: tl.constexpr, op: tl.constexpr,
    dtype: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    h = tl.load(h_ptr + offsets, mask=offsets < N)

    sg = op(h, dtype)
    tl.store(sg_ptr + offsets, sg, mask=offsets < N)


def sg_high_level_kernel_wrapper(h: torch.Tensor, op: triton.JITFunction):
    sg = torch.empty_like(h)
    N = h.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sg_high_level_kernel[grid](h, sg, N, BLOCK_SIZE, op, type_dict[h.dtype])
    return sg


@pytest.mark.parametrize("shape", SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_sigmoid_sg_torch2triton(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    sg1 = sg_high_level_kernel_wrapper(
        x, surrogate_kernels.sigmoid_surrogate_backward
    )
    sigmoid_surrogate_t2t = torch2triton.transpile_triton_code(
        sigmoid_surrogate_torch, verbose=True
    )
    sg2 = sg_high_level_kernel_wrapper(x, sigmoid_surrogate_t2t)
    assert_close(sg1, sg2, prefix="sg_sigmoid")


def lif_core(x: torch.Tensor, v: torch.Tensor, beta: torch.Tensor):
    h = v*beta + x
    s = torch2triton.spike_fn(h, 1.)
    v = h * (1.-s)
    return s, v


@triton.jit
def _multistep_lif_high_level_inference_kernel(
    x_seq_ptr,  # [T, NCL]
    s_seq_ptr,
    beta,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    op: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v = tl.zeros([BLOCK_NCL], dtype=dtype)
    beta = tl.full([1], beta, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)

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

        s, v = op(x, v, beta)

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))


def multistep_lif_high_level_inference_kernel_wrapper(
    x_seq: torch.Tensor, beta: float, op: triton.JITFunction
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_NCL = 256
    s_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    print(x_seq.dtype, s_seq.dtype)
    _multistep_lif_high_level_inference_kernel[grid](
        x_seq,
        s_seq,
        beta,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
        op=op,
    )
    return s_seq


@pytest.mark.parametrize("shape", SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("beta", BETA_LIST)
def test_lif_torch2triton(shape, dtype, beta):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    s1 = lif.multistep_lif_hard_inference(x, beta)
    core = torch2triton.transpile_triton_code(lif_core, verbose=True)
    s2 = multistep_lif_high_level_inference_kernel_wrapper(x, beta, core)
    assert_close(s1, s2, prefix="lif_spike")


if __name__ == "__main__":

    def t(x):
        return torch2triton.spike_fn(x, 1.)

    print(torch.fx.symbolic_trace(t).graph)
