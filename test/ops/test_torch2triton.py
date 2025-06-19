import pytest
import sys

sys.path.append("./")

import torch
import triton
import triton.language as tl

from flashsnn.ops import torch2triton
from flashsnn.ops import surrogate_kernels
from flashsnn.utils import assert_close, type_dict

SHAPE_LIST = [(32, 3, 224, 224), (23, 700)]
DTYPE_LIST = [torch.float32, torch.float16]


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
def test_sigmoid_sg(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    sg1 = sg_high_level_kernel_wrapper(
        x, surrogate_kernels.sigmoid_surrogate_backward
    )
    sigmoid_surrogate_t2t = torch2triton.transpile_triton_code(
        sigmoid_surrogate_torch, verbose=True
    )
    sg2 = sg_high_level_kernel_wrapper(x, sigmoid_surrogate_t2t)
    assert_close(sg1, sg2, prefix="sg_sigmoid")


def lif_core(x: torch.Tensor, v: torch.Tensor, beta: float):
    h = v*beta + x
    s = (h >= 1.).to(torch.float32)
    v = h * (1.-s)
    return s, v
