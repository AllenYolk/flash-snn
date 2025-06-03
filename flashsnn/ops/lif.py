import torch
import triton
import triton.language as tl

from flashsnn.utils import type_dict


@triton.jit
def _multistep_lif_forward_hard_kernel(
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
            block_shape=(BLOCK_NCL, ncl_offset),
            order=(1, 0)
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        h = beta*v + x  # decay_input = False
        s = (h >= 1.).to(dtype)  # v_th = 1
        v = h * (one-s)  # hard_reset, v_reset = 0

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, pid_ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        h_ptrs = tl.make_block_ptr(
            h_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, pid_ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0)
        )
        tl.store(s_ptrs, s, boundary_check=(1,))
        tl.store(h_ptrs, h, boundary_check=(1,))


@triton.jit
def _multistep_lif_backward_atan_hard_not_detached_kernel(
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
        h_ptrs = tl.make_block_ptr(
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
        grad_v = (grad_s - grad_v*h) * sg + grad_v * (one-s)

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
def _multistep_lif_backward_atan_hard_detached_kernel(
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
        h_ptrs = tl.make_block_ptr(
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
        grad_v = grad_s*sg + grad_v * (one-s)

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


def multistep_lif_forward_hard(x_seq: torch.Tensor, beta: float):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    s_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    BLOCK_NCL = 128
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_forward_hard_kernel[grid](
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


def multistep_lif_backward_not_detached_triton(
    grad_s_seq: torch.Tensor, h_seq: torch.Tensor, s_seq: torch.Tensor,
    beta: float
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = torch.empty_like(grad_s_seq)
    dtype = grad_s_seq.dtype
    BLOCK_NCL = 128
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_backward_atan_hard_not_detached_kernel[grid](
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


def multistep_lif_backward_detached_triton(
    grad_s_seq: torch.Tensor, h_seq: torch.Tensor, s_seq: torch.Tensor,
    beta: float
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    grad_x_seq = torch.empty_like(grad_s_seq)
    dtype = grad_s_seq.dtype
    BLOCK_NCL = 128
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    _multistep_lif_backward_atan_hard_detached_kernel[grid](
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
