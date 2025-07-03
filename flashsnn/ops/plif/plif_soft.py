from functools import lru_cache
from typing import Callable

import torch
from torch import autograd
import triton
import triton.language as tl

from flashsnn.ops import surrogate_kernels
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
    restore_value=["s_seq_ptr"],
)
@triton.jit
def _multistep_plif_soft_inference_kernel(
    x_seq_ptr,  # [T, NCL]
    beta_seq_ptr,  # [T, NCL], before applying sigmoid
    s_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v = tl.zeros([1, BLOCK_NCL], dtype=dtype)

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

        h = tl.fma(beta, v, x)  # decay_input = False
        s = (h >= 1.).to(dtype)  # v_th = 1
        v = h - s  # soft_reset, v_th = 1

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype"],
    restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"],
)
@triton.jit
def _multistep_plif_soft_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    beta_seq_ptr,  # [T, NCL], before applying sigmoid
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v = tl.zeros([1, BLOCK_NCL], dtype=dtype)

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

        h = tl.fma(beta, v, x)
        s = (h >= 1.).to(dtype)  # v_th = 1
        v = h - s  # soft_reset, v_th = 1

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))
        tl.store(h_ptrs, h, boundary_check=(1,))
        tl.store(v_ptrs, v, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype"],
    restore_value=["grad_x_seq_ptr", "grad_beta_seq_ptr"],
)
@triton.jit
def _multistep_plif_soft_not_detached_backward_kernel(
    grad_s_seq_ptr,
    beta_seq_ptr,  # before applying sigmoid
    h_seq_ptr,
    v_seq_ptr,
    grad_x_seq_ptr,
    grad_beta_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    sg_fn: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([1, BLOCK_NCL], dtype=dtype)

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
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero")
        v_last_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t - 1, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        v_last = tl.load(
            v_last_ptrs, boundary_check=(0, 1), padding_option="zero"
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

        sg = sg_fn(h - 1.)
        grad_v = tl.fma(grad_s - grad_v, sg, grad_v)

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_v, boundary_check=(1,))

        grad_beta = grad_v * v_last * _sigmoid_backward(beta)
        grad_beta_ptrs = tl.make_block_ptr(
            grad_beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_beta_ptrs, grad_beta, boundary_check=(1,))

        grad_v = grad_v * beta


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype"],
    restore_value=["grad_x_seq_ptr", "grad_beta_seq_ptr"],
)
@triton.jit
def _multistep_plif_soft_detached_backward_kernel(
    grad_s_seq_ptr,
    beta_seq_ptr,  # before applying sigmoid
    h_seq_ptr,
    v_seq_ptr,
    grad_x_seq_ptr,
    grad_beta_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    sg_fn: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([1, BLOCK_NCL], dtype=dtype)

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
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        h = tl.load(h_ptrs, boundary_check=(1,), padding_option="zero")
        v_last_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t - 1, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        v_last = tl.load(
            v_last_ptrs, boundary_check=(0, 1), padding_option="zero"
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

        sg = sg_fn(h - 1.)
        grad_v = tl.fma(grad_s, sg, grad_v)

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_v, boundary_check=(1,))

        grad_beta = grad_v * v_last * _sigmoid_backward(beta)
        grad_beta_ptrs = tl.make_block_ptr(
            grad_beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_beta_ptrs, grad_beta, boundary_check=(1,))

        grad_v = grad_v * beta


def multistep_plif_soft_inference(
    x_seq: torch.Tensor, beta: torch.Tensor, inplace: bool = False
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = x_seq if inplace else torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_soft_inference_kernel[grid](
        x_seq,
        beta,
        s_seq,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return s_seq


def multistep_plif_soft_forward(
    x_seq: torch.Tensor, beta: torch.Tensor, inplace: bool = False
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = x_seq if inplace else torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_soft_forward_kernel[grid](
        x_seq,
        beta,
        s_seq,
        h_seq,
        v_seq,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return s_seq, h_seq, v_seq


def multistep_plif_soft_not_detached_backward(
    grad_s_seq: torch.Tensor,
    beta: torch.Tensor,
    h_seq: torch.Tensor,
    v_seq: torch.Tensor,
    sg_fn: Callable,
    inplace: bool = False
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = grad_s_seq if inplace else torch.empty_like(grad_s_seq)
    grad_beta = torch.empty_like(beta)
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_soft_not_detached_backward_kernel[grid](
        grad_s_seq,
        beta,
        h_seq,
        v_seq,
        grad_x_seq,
        grad_beta,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        sg_fn=sg_fn,
    )
    return grad_x_seq, grad_beta


def multistep_plif_soft_detached_backward(
    grad_s_seq: torch.Tensor,
    beta: torch.Tensor,
    h_seq: torch.Tensor,
    v_seq: torch.Tensor,
    sg_fn: Callable,
    inplace: bool = False
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = grad_s_seq if inplace else torch.empty_like(grad_s_seq)
    grad_beta = torch.empty_like(beta)
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_soft_detached_backward_kernel[grid](
        grad_s_seq,
        beta,
        h_seq,
        v_seq,
        grad_x_seq,
        grad_beta,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        sg_fn=sg_fn,
    )
    return grad_x_seq, grad_beta


class MultistepPLIFSoftNotDetachedFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx, x_seq: torch.Tensor, beta: torch.Tensor, sg_fn: Callable,
        fwd_inplace: bool, bwd_inplace: bool
    ):
        # beta: after applying sigmoid
        if any(ctx.needs_input_grad):
            s_seq, h_seq, v_seq = multistep_plif_soft_forward(
                x_seq, beta, fwd_inplace
            )
            ctx.save_for_backward(h_seq, v_seq, beta)
            ctx.sg_fn = sg_fn
            ctx.bwd_inplace = bwd_inplace
        else:
            s_seq = multistep_plif_soft_inference(x_seq, beta, fwd_inplace)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq, v_seq, beta = ctx.saved_tensors
        grad_x_seq, grad_beta = multistep_plif_soft_not_detached_backward(
            grad_s_seq, beta, h_seq, v_seq, ctx.sg_fn, ctx.bwd_inplace
        )
        return grad_x_seq, grad_beta, None, None, None


class MultistepPLIFSoftDetachedFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx, x_seq: torch.Tensor, beta: float, sg_fn: Callable,
        fwd_inplace: bool, bwd_inplace: bool
    ):
        # beta: after applying sigmoid
        if any(ctx.needs_input_grad):
            s_seq, h_seq, v_seq = multistep_plif_soft_forward(
                x_seq, beta, fwd_inplace
            )
            ctx.save_for_backward(h_seq, v_seq, beta)
            ctx.sg_fn = sg_fn
            ctx.bwd_inplace = bwd_inplace
        else:
            s_seq = multistep_plif_soft_inference(x_seq, beta, fwd_inplace)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq, v_seq, beta = ctx.saved_tensors
        grad_x_seq, grad_beta = multistep_plif_soft_detached_backward(
            grad_s_seq, beta, h_seq, v_seq, ctx.sg_fn, ctx.bwd_inplace
        )
        return grad_x_seq, grad_beta, None, None, None
