from typing import Callable

import torch
from torch import autograd
import triton
import triton.language as tl

from flashsnn.utils import type_dict, contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype", "soft_reset", "save_intermediates"],
    restore_value=["s_seq_ptr"],  # if inplace, we must restore s_seq
)
@triton.jit
def _multistep_lif_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    s_seq_ptr,
    h_seq_ptr,
    beta,
    vth,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v = tl.zeros([1, BLOCK_NCL], dtype=dtype)
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

        h = tl.fma(beta, v, x)
        s = (h >= vth).to(dtype)
        if soft_reset:
            v = h - s
        else:
            v = h * (1.-s)  # hard_reset, v_reset = 0

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0)
            )
            tl.store(h_ptrs, h, boundary_check=(1,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2, 4]
        for w in [2, 4, 8]
    ],
    key=["T", "dtype", "soft_reset", "detach_reset"],
    restore_value=["grad_x_seq_ptr"],
)
@triton.jit
def _multistep_lif_backward_kernel(
    grad_s_seq_ptr,
    h_seq_ptr,
    grad_x_seq_ptr,
    beta,
    vth,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,  # grad_s_seq.dtype; might != h_seq or s_seq.dtype
    sg_fn: tl.constexpr,
    soft_reset: tl.constexpr,
    detach_reset: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([1, BLOCK_NCL], dtype=dtype)
    beta = tl.full([1], beta, dtype=dtype)

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

        sg = sg_fn(h - vth)
        if soft_reset:
            if detach_reset:
                grad_v = tl.fma(grad_s, sg, grad_v)
            else:
                grad_v = tl.fma(grad_s - grad_v, sg, grad_v)
        else:
            s = (h >= vth).to(dtype)
            if detach_reset:
                # grad_v = grad_s*sg + grad_v * (one-s)
                grad_v = tl.fma(grad_s, sg, grad_v * (1.-s))
            else:
                # grad_v = (grad_s - grad_v*h) * sg + grad_v * (one-s)
                grad_v = tl.fma(tl.fma(-grad_v, h, grad_s), sg, grad_v * (1.-s))

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_v.to(dtype), boundary_check=(1,))
        grad_v = grad_v * beta


def multistep_lif_inference(
    x_seq: torch.Tensor,
    beta: float,
    vth: float,
    soft_reset: bool,
    inplace: bool = False,
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = x_seq if inplace else torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_forward_kernel[grid](
        x_seq,
        s_seq,
        None,
        beta,
        vth,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        soft_reset=soft_reset,
        save_intermediates=False,
    )
    return s_seq


def multistep_lif_forward(
    x_seq: torch.Tensor,
    beta: float,
    vth: float,
    soft_reset: bool,
    inplace: bool = False,
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = x_seq if inplace else torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_forward_kernel[grid](
        x_seq,
        s_seq,
        h_seq,
        beta,
        vth,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        soft_reset=soft_reset,
        save_intermediates=True,
    )
    return s_seq, h_seq


def multistep_lif_backward(
    grad_s_seq: torch.Tensor,
    h_seq: torch.Tensor,
    beta: float,
    vth: float,
    sg_fn: Callable,
    soft_reset: bool,
    detach_reset: bool,
    inplace: bool = False
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = grad_s_seq if inplace else torch.empty_like(grad_s_seq)
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_backward_kernel[grid](
        grad_s_seq,
        h_seq,
        grad_x_seq,
        beta,
        vth,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
        sg_fn=sg_fn,
        soft_reset=soft_reset,
        detach_reset=detach_reset,
    )
    return grad_x_seq


class MultistepLIFFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx, x_seq: torch.Tensor, beta: float, vth: float, sg_fn: Callable,
        soft_reset: bool, detach_reset: bool, fwd_inplace: bool,
        bwd_inplace: bool
    ):
        if any(ctx.needs_input_grad):
            s_seq, h_seq = multistep_lif_forward(
                x_seq, beta, vth, soft_reset, fwd_inplace
            )
            ctx.save_for_backward(h_seq)
            ctx.beta = beta
            ctx.vth = vth
            ctx.sg_fn = sg_fn
            ctx.soft_reset = soft_reset
            ctx.detach_reset = detach_reset
            ctx.bwd_inplace = bwd_inplace
        else:
            s_seq = multistep_lif_inference(
                x_seq, beta, vth, soft_reset, fwd_inplace
            )
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq = ctx.saved_tensors[0]
        grad_x_seq = multistep_lif_backward(
            grad_s_seq, h_seq, ctx.beta, ctx.vth, ctx.sg_fn, ctx.soft_reset,
            ctx.detach_reset, ctx.bwd_inplace
        )
        return grad_x_seq, None, None, None, None, None, None, None
