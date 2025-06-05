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
    BLOCK_NCL = min(128, max(1024, BLOCK_NCL))
    return BLOCK_NCL


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [2, 4, 8]],
    key=["T", "NCL", "BLOCK_NCL", "dtype"],
)
@triton.jit
def _multistep_plif_hard_inference_kernel(
    x_seq_ptr,  # [T, N, C, L]
    beta_seq_ptr,  # [T, N, C, L], after applying sigmoid
    s_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v = tl.zeros([BLOCK_NCL], dtype=dtype)
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
        beta_ptrs = tl.make_block_ptr(
            beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        beta = tl.load(beta_ptrs, boundary_check=(1,), padding_option="zero")

        h = beta*v + x  # decay_input = False
        s = (h >= 1.).to(dtype)  # v_th = 1
        v = h * (one-s)  # hard_reset, v_reset = 0

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
    configs=[triton.Config({}, num_warps=w) for w in [2, 4, 8]],
    key=["T", "NCL", "BLOCK_NCL", "dtype"],
)
@triton.jit
def _multistep_plif_hard_forward_kernel(
    x_seq_ptr,  # [T, N, C, L]
    beta_seq_ptr,  # [T, N, C, L], after applying sigmoid
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

    v = tl.zeros([BLOCK_NCL], dtype=dtype)
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
        beta_ptrs = tl.make_block_ptr(
            beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        beta = tl.load(beta_ptrs, boundary_check=(1,), padding_option="zero")

        h = beta*v + x  # decay_input = False
        s = (h >= 1.).to(dtype)  # v_th = 1
        v = h * (one-s)  # hard_reset, v_reset = 0

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
    configs=[triton.Config({}, num_warps=w) for w in [2, 4, 8]],
    key=["T", "NCL", "BLOCK_NCL", "dtype"],
)
@triton.jit
def _multistep_plif_hard_atan_not_detached_backward_kernel(
    grad_s_seq_ptr,
    beta_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    s_seq_ptr,
    grad_x_seq_ptr,
    grad_beta_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
    pi = tl.full([1], 3.141592653589793, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)

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
        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        s = tl.load(s_ptrs, boundary_check=(1,), padding_option="zero")
        beta_ptrs = tl.make_block_ptr(
            beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        beta = tl.load(beta_ptrs, boundary_check=(1,), padding_option="zero")

        sg = pi * (h-one)
        sg = (one / (one + sg*sg)).to(dtype)
        grad_v = (grad_s - grad_v*h) * sg + grad_v * (one-s)  # grad_h

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_v, boundary_check=(1,))

        grad_beta = grad_v * v_last
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
    configs=[triton.Config({}, num_warps=w) for w in [2, 4, 8]],
    key=["T", "NCL", "BLOCK_NCL", "dtype"],
)
@triton.jit
def _multistep_plif_hard_atan_detached_backward_kernel(
    grad_s_seq_ptr,
    beta_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    s_seq_ptr,
    grad_x_seq_ptr,
    grad_beta_seq_ptr,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
    pi = tl.full([1], 3.141592653589793, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)

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
        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        s = tl.load(s_ptrs, boundary_check=(1,), padding_option="zero")
        beta_ptrs = tl.make_block_ptr(
            beta_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        beta = tl.load(beta_ptrs, boundary_check=(1,), padding_option="zero")

        sg = pi * (h-one)
        sg = (one / (one + sg*sg)).to(dtype)
        grad_v = grad_s*sg + grad_v * (one-s)  # grad_h

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_v, boundary_check=(1,))

        grad_beta = grad_v * v_last
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


def multistep_plif_hard_inference(x_seq: torch.Tensor, beta: torch.Tensor):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, x_seq.device.index)
    s_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_hard_inference_kernel[grid](
        x_seq,
        beta,
        s_seq,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return s_seq


def multistep_plif_hard_forward(x_seq: torch.Tensor, beta: torch.Tensor):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, x_seq.device.index)
    s_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    v_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_hard_forward_kernel[grid](
        x_seq,
        beta,
        s_seq,
        h_seq,
        v_seq,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return s_seq, h_seq, v_seq


def multistep_plif_hard_atan_not_detached_backward(
    grad_s_seq: torch.Tensor,
    beta: torch.Tensor,
    h_seq: torch.Tensor,
    v_seq: torch.Tensor,
    s_seq: torch.Tensor,
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, grad_s_seq.device.index)
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_beta = torch.empty_like(beta)
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_hard_atan_not_detached_backward_kernel[grid](
        grad_s_seq,
        beta,
        h_seq,
        v_seq,
        s_seq,
        grad_x_seq,
        grad_beta,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq, grad_beta


def multistep_plif_hard_atan_detached_backward(
    grad_s_seq: torch.Tensor,
    beta: torch.Tensor,
    h_seq: torch.Tensor,
    v_seq: torch.Tensor,
    s_seq: torch.Tensor,
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, grad_s_seq.device.index)
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_beta = torch.empty_like(beta)
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_plif_hard_atan_detached_backward_kernel[grid](
        grad_s_seq,
        beta,
        h_seq,
        v_seq,
        s_seq,
        grad_x_seq,
        grad_beta,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq, grad_beta


class MultistepPLIFAtanHardNotDetachedFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, x_seq: torch.Tensor, beta: torch.Tensor):
        # beta: after applying sigmoid
        if any(ctx.needs_input_grad):
            s_seq, h_seq, v_seq = multistep_plif_hard_forward(x_seq, beta)
            ctx.save_for_backward(h_seq, v_seq, s_seq, beta)
        else:
            s_seq = multistep_plif_hard_inference(x_seq, beta)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq, v_seq, s_seq, beta = ctx.saved_tensors
        grad_x_seq, grad_beta = multistep_plif_hard_atan_not_detached_backward(
            grad_s_seq, beta, h_seq, v_seq, s_seq
        )
        return grad_x_seq, grad_beta


class MultistepPLIFAtanHardDetachedFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, x_seq: torch.Tensor, beta: float):
        # beta: after applying sigmoid
        if any(ctx.needs_input_grad):
            s_seq, h_seq, v_seq = multistep_plif_hard_forward(x_seq, beta)
            ctx.save_for_backward(h_seq, v_seq, s_seq, beta)
        else:
            s_seq = multistep_plif_hard_inference(x_seq, beta)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq, v_seq, s_seq, beta = ctx.saved_tensors
        grad_x_seq, grad_beta = multistep_plif_hard_atan_detached_backward(
            grad_s_seq, beta, h_seq, v_seq, s_seq
        )
        return grad_x_seq, grad_beta
