from functools import lru_cache

import torch
from torch import autograd
import triton
import triton.language as tl

from flashsnn.utils import type_dict, contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd


@triton.jit
def _sigmoid_forward(x, dtype: tl.constexpr):
    return tl.sigmoid(x.to(tl.float32)).to(dtype)


@triton.jit
def _sigmoid_backward(y):
    # y = sigmoid(x)
    y = y * (1.-y)
    return y


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype"],
    restore_value=["y_seq_ptr"],
)
@triton.jit
def _multistep_pli_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    beta_seq_ptr,  # [T, NCL], before applying sigmoid
    y_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    y = tl.zeros([1, BLOCK_NCL], dtype=dtype)

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
        beta_ptrs = tl.make_block_ptr(
            beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        beta = tl.load(beta_ptrs, boundary_check=(1,), padding_option="zero")
        beta = _sigmoid_forward(beta, dtype)

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
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype"],
    restore_value=["grad_x_seq_ptr", "grad_beta_seq_ptr"]
)
@triton.jit
def _multistep_pli_backward_kernel(
    grad_y_seq_ptr,
    beta_seq_ptr,
    y_seq_ptr,
    grad_x_seq_ptr,
    grad_beta_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    dy = tl.zeros([1, BLOCK_NCL], dtype=dtype)

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
        y_last_ptrs = tl.make_block_ptr(
            y_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t - 1, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        y_last = tl.load(
            y_last_ptrs, boundary_check=(0, 1), padding_option="zero"
        )
        beta_ptrs = tl.make_block_ptr(
            beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        beta = tl.load(beta_ptrs, boundary_check=(1,), padding_option="zero")
        beta = _sigmoid_forward(beta, dtype)

        dy = tl.fma(beta, dy, grad_y)
        d_beta = dy * y_last * _sigmoid_backward(beta)

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, dy, boundary_check=(1,))
        grad_beta_ptrs = tl.make_block_ptr(
            grad_beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_beta_ptrs, d_beta, boundary_check=(1,))


def multistep_pli_forward(
    x_seq: torch.Tensor, beta: torch.Tensor, inplace: bool = False
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    y_seq = x_seq if inplace else torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_pli_forward_kernel[grid](
        x_seq,
        beta,
        y_seq,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return y_seq


def multistep_pli_backward(
    grad_y_seq: torch.Tensor,
    beta: torch.Tensor,
    y_seq: torch.Tensor,
    inplace: bool = False
):
    T = grad_y_seq.shape[0]
    NCL = grad_y_seq[0].numel()
    grad_x_seq = grad_y_seq if inplace else torch.empty_like(grad_y_seq)
    grad_beta = torch.empty_like(beta)
    dtype = grad_y_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_pli_backward_kernel[grid](
        grad_y_seq,
        beta,
        y_seq,
        grad_x_seq,
        grad_beta,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq, grad_beta


class MultistepPLIFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx, x_seq: torch.Tensor, beta: torch.Tensor, fwd_inplace: bool,
        bwd_inplace: bool
    ):
        """beta.shape=[T, NCL]; after applying sigmoid"""
        y_seq = multistep_pli_forward(x_seq, beta, fwd_inplace)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(y_seq, beta)
            ctx.bwd_inplace = bwd_inplace
        return y_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_y_seq: torch.Tensor):
        y_seq, beta = ctx.saved_tensors
        grad_x_seq, grad_beta = multistep_pli_backward(
            grad_y_seq, beta, y_seq, ctx.bwd_inplace
        )
        return grad_x_seq, grad_beta, None, None
