import torch
from torch import autograd
import triton
import triton.language as tl

from flashsnn.utils import type_dict, contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd


@triton.jit
def _multistep_lif_soft_inference_kernel(
    x_seq_ptr,  # [T, N, C, L]
    s_seq_ptr,
    beta,
    T: tl.constexpr,
    NCL,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
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

        h = beta*v + x  # decay_input = False
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


@triton.jit
def _multistep_lif_soft_forward_kernel(
    x_seq_ptr,  # [T, N, C, L]
    s_seq_ptr,
    h_seq_ptr,
    beta,
    T: tl.constexpr,
    NCL,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
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

        h = beta*v + x  # decay_input = False
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
        tl.store(s_ptrs, s, boundary_check=(1,))
        tl.store(h_ptrs, h, boundary_check=(1,))


@triton.jit
def _multistep_lif_soft_atan_not_detached_backward_kernel(
    grad_s_seq_ptr,
    h_seq_ptr,
    s_seq_ptr,
    grad_x_seq_ptr,
    beta,
    T: tl.constexpr,
    NCL,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
    pi = tl.full([1], 3.141592653589793, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)
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
        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        s = tl.load(s_ptrs, boundary_check=(1,), padding_option="zero")

        sg = pi * (h-one)
        sg = (one / (one + sg*sg)).to(dtype)
        grad_v = (grad_s-grad_v) * sg + grad_v

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_v, boundary_check=(1,))
        grad_v = grad_v * beta


@triton.jit
def _multistep_lif_soft_atan_detached_backward_kernel(
    grad_s_seq_ptr,
    h_seq_ptr,
    s_seq_ptr,
    grad_x_seq_ptr,
    beta,
    T: tl.constexpr,
    NCL,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
    pi = tl.full([1], 3.141592653589793, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)
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
        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        s = tl.load(s_ptrs, boundary_check=(1,), padding_option="zero")

        sg = pi * (h-one)
        sg = (one / (one + sg*sg)).to(dtype)
        grad_v = grad_s*sg + grad_v

        grad_x_ptrs = tl.make_block_ptr(
            grad_x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(grad_x_ptrs, grad_v, boundary_check=(1,))
        grad_v = grad_v * beta


def multistep_lif_soft_inference(x_seq: torch.Tensor, beta: float):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    BLOCK_NCL = 128
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_soft_inference_kernel[grid](
        x_seq,
        s_seq,
        beta,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return s_seq


def multistep_lif_soft_forward(x_seq: torch.Tensor, beta: float):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    BLOCK_NCL = 128
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_soft_forward_kernel[grid](
        x_seq,
        s_seq,
        h_seq,
        beta,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return s_seq, h_seq


def multistep_lif_soft_atan_not_detached_backward(
    grad_s_seq: torch.Tensor, h_seq: torch.Tensor, s_seq: torch.Tensor,
    beta: float
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = torch.empty_like(grad_s_seq)
    dtype = grad_s_seq.dtype
    BLOCK_NCL = 128
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_soft_atan_not_detached_backward_kernel[grid](
        grad_s_seq,
        h_seq,
        s_seq,
        grad_x_seq,
        beta,
        T,
        NCL,
        BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq


def multistep_lif_soft_atan_detached_backward(
    grad_s_seq: torch.Tensor, h_seq: torch.Tensor, s_seq: torch.Tensor,
    beta: float
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = torch.empty_like(grad_s_seq)
    dtype = grad_s_seq.dtype
    BLOCK_NCL = 128
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_soft_atan_detached_backward_kernel[grid](
        grad_s_seq,
        h_seq,
        s_seq,
        grad_x_seq,
        beta,
        T,
        NCL,
        BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq


class MultistepLIFAtanSoftNotDetachedFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, x_seq: torch.Tensor, beta: float):
        if any(ctx.needs_input_grad):
            s_seq, h_seq = multistep_lif_soft_forward(x_seq, beta)
            ctx.save_for_backward(h_seq, s_seq)
            ctx.beta = beta
        else:
            s_seq = multistep_lif_soft_inference(x_seq, beta)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq, s_seq = ctx.saved_tensors
        grad_x_seq = multistep_lif_soft_atan_not_detached_backward(
            grad_s_seq, h_seq, s_seq, ctx.beta
        )
        return grad_x_seq, None


class MultistepLIFAtanSoftDetachedFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, x_seq: torch.Tensor, beta: float):
        if any(ctx.needs_input_grad):
            s_seq, h_seq = multistep_lif_soft_forward(x_seq, beta)
            ctx.save_for_backward(h_seq, s_seq)
            ctx.beta = beta
        else:
            s_seq = multistep_lif_soft_inference(x_seq, beta)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        h_seq, s_seq = ctx.saved_tensors
        grad_x_seq = multistep_lif_soft_atan_detached_backward(
            grad_s_seq, h_seq, s_seq, ctx.beta
        )
        return grad_x_seq, None
