from functools import lru_cache

import torch
from torch import autograd
import triton
import triton.language as tl

from flashsnn.utils import get_multiprocessor_count, type_dict
from flashsnn.utils import contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd


@lru_cache(maxsize=None)
def _get_block_size(T, NCL, device_idx):
    BLOCK_T = triton.next_power_of_2(T)
    BLOCK_T = max(16, BLOCK_T)  # BLOCK_T >= T, BLOCK_T >= 16
    BLOCK_NCL = triton.next_power_of_2(
        triton.cdiv(NCL, get_multiprocessor_count(device_idx))
    )
    BLOCK_NCL = min(256, max(32, BLOCK_NCL))
    return BLOCK_T, BLOCK_NCL


@triton.jit
def _psn_inference_kernel(
    x_seq_ptr,  # [T, N, C, L]
    weight_ptr,  # [T, T]
    bias_ptr,  # [T, 1]
    s_seq_ptr,  # [T, N, C, L]
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_T: tl.constexpr,  # >= T
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    x_ptrs = tl.make_block_ptr(
        x_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    x_seq = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")
    weight_ptrs = tl.make_block_ptr(
        weight_ptr,
        shape=(T, T),
        strides=(T, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_T, BLOCK_T),
        order=(1, 0)
    )
    weight = tl.load(weight_ptrs, boundary_check=(0, 1), padding_option="zero")
    bias_ptrs = tl.make_block_ptr(
        bias_ptr,
        shape=(T, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_T, 1),
        order=(1, 0)
    )
    bias = tl.load(bias_ptrs, boundary_check=(0,), padding_option="zero")

    x_seq = tl.dot(weight, x_seq)
    h_seq = x_seq - bias
    s_seq = (h_seq >= 0.).to(dtype)

    s_ptrs = tl.make_block_ptr(
        s_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    tl.store(s_ptrs, s_seq, boundary_check=(0, 1))


@triton.jit
def _psn_forward_kernel(
    x_seq_ptr,  # [T, N, C, L]
    weight_ptr,  # [T, T]
    bias_ptr,  # [T, 1]
    s_seq_ptr,  # [T, N, C, L]
    h_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    x_ptrs = tl.make_block_ptr(
        x_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    x_seq = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")
    weight_ptrs = tl.make_block_ptr(
        weight_ptr,
        shape=(T, T),
        strides=(T, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_T, BLOCK_T),
        order=(1, 0)
    )
    weight = tl.load(weight_ptrs, boundary_check=(0, 1), padding_option="zero")
    bias_ptrs = tl.make_block_ptr(
        bias_ptr,
        shape=(T, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_T, 1),
        order=(1, 0)
    )
    bias = tl.load(bias_ptrs, boundary_check=(0,), padding_option="zero")

    x_seq = tl.dot(weight, x_seq)
    h_seq = x_seq - bias
    s_seq = (h_seq >= 0.).to(dtype)

    s_ptrs = tl.make_block_ptr(
        s_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    tl.store(s_ptrs, s_seq, boundary_check=(0, 1))
    h_ptrs = tl.make_block_ptr(
        h_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    tl.store(h_ptrs, h_seq, boundary_check=(0, 1))


@triton.jit
def _psn_atan_backward_kernel(
    grad_s_seq_ptr,  # [T, N, C, L]
    weight_ptr,  # [T, T]
    h_seq_ptr,
    x_seq_ptr,
    grad_x_seq_ptr,
    grad_weight_ptr,  # [T, T]
    grad_bias_ptr,  # [T, 1]
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    pi = tl.full([1], 3.141592653589793, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)

    grad_s_seq_ptrs = tl.make_block_ptr(
        grad_s_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    grad_s_seq = tl.load(
        grad_s_seq_ptrs, boundary_check=(0, 1), padding_option="zero"
    )
    weight_ptrs = tl.make_block_ptr(
        weight_ptr,
        shape=(T, T),
        strides=(T, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_T, BLOCK_T),
        order=(1, 0)
    )
    weight = tl.load(weight_ptrs, boundary_check=(0, 1), padding_option="zero")
    h_ptrs = tl.make_block_ptr(
        h_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    h_seq = tl.load(h_ptrs, boundary_check=(0, 1), padding_option="zero")
    x_ptrs = tl.make_block_ptr(
        x_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    x_seq = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")

    sg = pi * h_seq
    sg = (one / (one + sg*sg)).to(dtype)
    grad_h_seq = grad_s_seq * sg  # [T, NCL]
    grad_x_seq = tl.dot(tl.trans(weight), grad_h_seq)
    grad_weight = tl.dot(grad_h_seq, tl.trans(x_seq))
    grad_bias = -tl.sum(grad_h_seq, axis=1, keep_dims=True)

    grad_x_seq_ptrs = tl.make_block_ptr(
        grad_x_seq_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(BLOCK_T, BLOCK_NCL),
        order=(1, 0)
    )
    tl.store(
        grad_x_seq_ptrs,
        grad_x_seq,
        boundary_check=(0, 1),
    )
    grad_weight_ptrs = tl.make_block_ptr(
        grad_weight_ptr,
        shape=(T, T),
        strides=(T, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_T, BLOCK_T),
        order=(1, 0)
    )
    tl.store(
        grad_weight_ptrs,
        grad_weight,
        boundary_check=(0, 1),
    )
    grad_bias_ptrs = tl.make_block_ptr(
        grad_bias_ptr,
        shape=(T, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_T, 1),
        order=(1, 0)
    )
    tl.store(
        grad_bias_ptrs,
        grad_bias,
        boundary_check=(0,),
    )


def psn_inference(
    x_seq: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_T, BLOCK_NCL = _get_block_size(T, NCL, x_seq.device.index)
    s_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _psn_inference_kernel[grid](
        x_seq,
        weight,
        bias,
        s_seq,
        T=T,
        NCL=NCL,
        BLOCK_T=BLOCK_T,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return s_seq


def psn_forward(x_seq: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_T, BLOCK_NCL = _get_block_size(T, NCL, x_seq.device.index)
    s_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _psn_forward_kernel[grid](
        x_seq,
        weight,
        bias,
        s_seq,
        h_seq,
        T=T,
        NCL=NCL,
        BLOCK_T=BLOCK_T,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return s_seq, h_seq


def psn_atan_backward(
    grad_s_seq: torch.Tensor, weight: torch.Tensor, h_seq: torch.Tensor,
    x_seq: torch.Tensor
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    BLOCK_T, BLOCK_NCL = _get_block_size(T, NCL, grad_s_seq.device.index)
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty_like(weight[0].unsqueeze(1))  # shape=[T, 1]
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _psn_atan_backward_kernel[grid](
        grad_s_seq,
        weight,
        h_seq,
        x_seq,
        grad_x_seq,
        grad_weight,
        grad_bias,
        T=T,
        NCL=NCL,
        BLOCK_T=BLOCK_T,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq, grad_weight, grad_bias


class PSNAtanFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx, x_seq: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ):
        if any(ctx.needs_input_grad):
            s_seq, h_seq = psn_forward(x_seq, weight, bias)
            ctx.save_for_backward(h_seq, x_seq, weight)
        else:
            s_seq = psn_inference(x_seq, weight, bias)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq, x_seq, weight = ctx.saved_tensors
        grad_x_seq, grad_weight, grad_bias = psn_atan_backward(
            grad_s_seq, weight, h_seq, x_seq
        )
        return grad_x_seq, grad_weight, grad_bias
