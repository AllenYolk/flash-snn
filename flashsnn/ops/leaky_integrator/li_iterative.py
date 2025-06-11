from functools import lru_cache

import torch
from torch import autograd
import triton
import triton.language as tl

from flashsnn.utils import type_dict, contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd
from flashsnn.utils import get_multiprocessor_count


@lru_cache(maxsize=None)
def _get_block_size(NCL, device_idx):
    BLOCK_NCL = triton.next_power_of_2(
        triton.cdiv(NCL, get_multiprocessor_count(device_idx))
    )
    BLOCK_NCL = min(1024, max(128, BLOCK_NCL))
    return BLOCK_NCL


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=["T", "BLOCK_NCL", "dtype"],
)
@triton.jit
def _multistep_li_forward_iterative_kernel(
    x_seq_ptr,  # [T, NCL]
    y_seq_ptr,
    beta,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    y = tl.zeros([BLOCK_NCL], dtype=dtype)
    beta = tl.full([1], beta, dtype=dtype)

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

        y = tl.fma(beta, y, x)  # fused element-wise multiply-add

        y_ptrs = tl.make_block_ptr(
            y_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(y_ptrs, y, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=["T", "BLOCK_NCL", "dtype"],
)
@triton.jit
def _multistep_li_backward_iterative_kernel(
    grad_y_seq_ptr,
    grad_x_seq_ptr,
    beta,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    dy = tl.zeros([BLOCK_NCL], dtype=dtype)
    beta = tl.full([1], beta, dtype=dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_y_ptrs = tl.make_block_ptr(
            grad_y_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        grad_y = tl.load(
            grad_y_ptrs, boundary_check=(1,), padding_option="zero"
        )

        dy = tl.fma(beta, dy, grad_y)

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, dy, boundary_check=(1,))


def multistep_li_forward_iterative(x_seq: torch.Tensor, beta: float):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, x_seq.device.index)
    y_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_li_forward_iterative_kernel[grid](
        x_seq,
        y_seq,
        beta,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return y_seq


def multistep_li_backward_iterative(grad_y_seq: torch.Tensor, beta: float):
    T = grad_y_seq.shape[0]
    NCL = grad_y_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, grad_y_seq.device.index)
    grad_x_seq = torch.empty_like(grad_y_seq)
    dtype = grad_y_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_li_backward_iterative_kernel[grid](
        grad_y_seq,
        grad_x_seq,
        beta,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq


class MultistepLIIterativeFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, x_seq: torch.Tensor, beta: float):
        y_seq = multistep_li_forward_iterative(x_seq, beta)
        ctx.beta = beta
        return y_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_y_seq: torch.Tensor):
        grad_x_seq = multistep_li_backward_iterative(grad_y_seq, ctx.beta)
        return grad_x_seq, None
