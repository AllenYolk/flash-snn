from typing import Callable

import torch
import torch.autograd as autograd
import triton
import triton.language as tl

from flashsnn.utils import type_dict
from flashsnn.utils import contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd
from flashsnn.utils import get_device_capability


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": b}, num_warps=w)
        for b in [16, 32, 64]
        for w in [4, 8, 16]
    ],
    key=["T", "N", "NUM_HEADS", "Cph", "L", "BLOCK_Cph"],
    restore_value=["output_ptr", "attn_lif_h_ptr"],
)
@triton.jit
def _token_qka_forward_kernel(
    qk_ptr,  # [T, N, 2, NUM_HEADS, Cph, L]
    output_ptr,  # [T, N, NUM_HEADS, Cph, L]
    attn_lif_h_ptr,  # [T, N, NUM_HEADS, 1, L]
    T: tl.constexpr,
    N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    Cph: tl.constexpr,
    L: tl.constexpr,
    BLOCK_Cph: tl.constexpr,  # >= Cph
    BLOCK_L: tl.constexpr,
    dtype: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    l_start = tl.program_id(2) * BLOCK_L

    c_stride = L
    h_stride = Cph * c_stride
    qk_stride = NUM_HEADS * h_stride
    n_stride = 2 * qk_stride
    t_stride = N * n_stride
    q_base = qk_ptr + n*n_stride + h*h_stride
    k_base = q_base + qk_stride
    output_t_stride = N * qk_stride
    output_base = output_ptr + n*qk_stride + h*h_stride
    if save_intermediates:
        h_t_stride = N * NUM_HEADS * L
        attn_lif_h_base = attn_lif_h_ptr + n*NUM_HEADS*L + h*L

    v = tl.zeros([1, BLOCK_L], dtype=dtype)
    beta = tl.full([1], 0.5, dtype=dtype)

    for t in tl.static_range(0, T, 1):
        q_ptrs = tl.make_block_ptr(
            q_base,
            shape=(Cph, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0)
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        k_ptrs = tl.make_block_ptr(
            k_base,
            shape=(Cph, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0)
        )
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")

        q = tl.sum(
            q.to(tl.float32),  #! avoid numerical issues
            axis=0,
            keep_dims=True,
        ).to(dtype)  # [1, BLOCK_L]; token-level
        # attn_lif: beta=0.5, vth=0.5, detach_reset=True
        h = tl.fma(beta, v, q)
        s = (h >= 0.5).to(dtype)  # [1, BLOCK_L]; token-level mask
        v = h * (1.-s)
        # apply the mask
        output = s * k

        output_ptrs = tl.make_block_ptr(
            output_base,
            shape=(Cph, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0)
        )
        tl.store(output_ptrs, output, boundary_check=(0, 1))
        if save_intermediates:
            attn_lif_h_ptrs = tl.make_block_ptr(
                attn_lif_h_base,
                shape=(1, L),
                strides=(c_stride, 1),
                offsets=(0, l_start),
                block_shape=(1, BLOCK_L),
                order=(1, 0)
            )
            tl.store(attn_lif_h_ptrs, h, boundary_check=(1,))

        q_base += t_stride
        k_base += t_stride
        output_base += output_t_stride
        if save_intermediates:
            attn_lif_h_base += h_t_stride


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": b}, num_warps=w)
        for b in [16, 32, 64]
        for w in [4, 8, 16]
    ],
    key=["T", "N", "NUM_HEADS", "Cph", "L", "BLOCK_Cph"],
    restore_value=["grad_qk_ptr"],
)
@triton.jit
def _token_qka_backward_kernel(
    grad_output_ptr,  # [T, N, NUM_HEADS, Cph, L]
    attn_lif_h_ptr,  # [T, N, NUM_HEADS, 1, L]
    qk_ptr,  # [T, N, 2, NUM_HEADS, Cph, L]
    grad_qk_ptr,  # [T, N, 2, NUM_HEADS, Cph, L]
    T: tl.constexpr,
    N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    Cph: tl.constexpr,
    L: tl.constexpr,
    BLOCK_Cph: tl.constexpr,  # >= Cph
    BLOCK_L: tl.constexpr,
    dtype: tl.constexpr,
    sg_fn: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    l_start = tl.program_id(2) * BLOCK_L

    c_stride = L
    h_stride = Cph * c_stride
    qk_stride = NUM_HEADS * h_stride
    n_stride = 2 * qk_stride
    t_stride = N * n_stride
    output_t_stride = N * qk_stride
    h_t_stride = N * NUM_HEADS * L

    grad_output_base = grad_output_ptr + n*qk_stride + h*h_stride
    attn_lif_h_base = attn_lif_h_ptr + n*NUM_HEADS*L + h*L
    k_base = qk_ptr + n*n_stride + h*h_stride + qk_stride
    grad_q_base = grad_qk_ptr + n*n_stride + h*h_stride
    grad_k_base = grad_q_base + qk_stride

    grad_output_base += (T-1) * output_t_stride
    attn_lif_h_base += (T-1) * h_t_stride
    k_base += (T-1) * t_stride
    grad_q_base += (T-1) * t_stride
    grad_k_base += (T-1) * t_stride

    grad_v = tl.zeros([1, BLOCK_L], dtype=dtype)
    beta = tl.full([1], 0.5, dtype=dtype)

    for t in range(T - 1, -1, -1):
        grad_output_ptrs = tl.make_block_ptr(
            grad_output_base,
            shape=(Cph, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0)
        )
        grad_output = tl.load(
            grad_output_ptrs, boundary_check=(0, 1), padding_option="zero"
        )
        attn_lif_h_ptrs = tl.make_block_ptr(
            attn_lif_h_base,
            shape=(1, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(1, BLOCK_L),
            order=(1, 0)
        )
        attn_lif_h = tl.load(
            attn_lif_h_ptrs, boundary_check=(1,), padding_option="zero"
        )
        s = (attn_lif_h >= 0.5).to(dtype)  # [1, BLOCK_L]; token-level mask
        k_ptrs = tl.make_block_ptr(
            k_base,
            shape=(Cph, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0)
        )
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")

        grad_k = grad_output * s
        grad_s = grad_output * k
        grad_s = tl.sum(grad_s.to(tl.float32), axis=0, keep_dims=True).to(dtype)
        # attn_lif: beta=0.5, vth=0.5, detach_reset=True
        sg = sg_fn(attn_lif_h - 0.5)
        # grad_v = grad_s*sg + grad_v * (one-s)
        grad_h = tl.fma(grad_s, sg, grad_v * (1.-s))
        grad_v = grad_h * beta
        grad_q = tl.broadcast_to(grad_h, [BLOCK_Cph, BLOCK_L])

        grad_q_ptrs = tl.make_block_ptr(
            grad_q_base,
            shape=(Cph, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0)
        )
        tl.store(grad_q_ptrs, grad_q, boundary_check=(0, 1))
        grad_k_ptrs = tl.make_block_ptr(
            grad_k_base,
            shape=(Cph, L),
            strides=(c_stride, 1),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0)
        )
        tl.store(grad_k_ptrs, grad_k, boundary_check=(0, 1))

        grad_output_base -= output_t_stride
        attn_lif_h_base -= h_t_stride
        k_base -= t_stride
        grad_q_base -= t_stride
        grad_k_base -= t_stride


def token_qka_inference(qk):
    # qk.shape = [T, N, 2, NUM_HEADS, Cph, L]
    T, N, _, NUM_HEADS, Cph, L = qk.shape
    BLOCK_Cph = max(triton.next_power_of_2(Cph), 16)

    output = torch.empty_like(qk[:, :, 0])
    grid = lambda meta: (N, NUM_HEADS, triton.cdiv(L, meta["BLOCK_L"]))
    _token_qka_forward_kernel[grid](
        qk,
        output,
        None,
        T,
        N,
        NUM_HEADS,
        Cph,
        L,
        BLOCK_Cph=BLOCK_Cph,
        dtype=type_dict[qk.dtype],
        save_intermediates=False,
    )
    return output


def token_qka_forward(qk):
    # qk.shape = [T, N, 2, NUM_HEADS, Cph, L]
    T, N, _, NUM_HEADS, Cph, L = qk.shape
    BLOCK_Cph = max(triton.next_power_of_2(Cph), 16)

    output = torch.empty_like(qk[:, :, 0])
    attn_lif_h = torch.empty_like(output[..., 0:1, :])
    grid = lambda meta: (N, NUM_HEADS, triton.cdiv(L, meta["BLOCK_L"]))
    _token_qka_forward_kernel[grid](
        qk,
        output,
        attn_lif_h,
        T,
        N,
        NUM_HEADS,
        Cph,
        L,
        BLOCK_Cph=BLOCK_Cph,
        dtype=type_dict[qk.dtype],
        save_intermediates=True,
    )
    return output, attn_lif_h


def token_qka_backward(grad_output, attn_lif_h, qk, sg_fn):
    # grad_output.shape = [T, N, NUM_HEADS, Cph, L]
    T, N, NUM_HEADS, Cph, L = grad_output.shape
    BLOCK_Cph = max(triton.next_power_of_2(Cph), 16)

    grad_qk = torch.empty_like(qk)
    grid = lambda meta: (N, NUM_HEADS, triton.cdiv(L, meta["BLOCK_L"]))
    _token_qka_backward_kernel[grid](
        grad_output,
        attn_lif_h,
        qk,
        grad_qk,
        T,
        N,
        NUM_HEADS,
        Cph,
        L,
        BLOCK_Cph=BLOCK_Cph,
        dtype=type_dict[qk.dtype],
        sg_fn=sg_fn,
    )
    return grad_qk


class TokenQKAFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, qk: torch.Tensor, sg_fn: Callable):
        if any(ctx.needs_input_grad):
            o, h = token_qka_forward(qk)
            ctx.save_for_backward(qk, h)
            ctx.sg_fn = sg_fn
        else:
            o = token_qka_inference(qk)
        return o

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        qk, h = ctx.saved_tensors
        grad_qk = token_qka_backward(grad_output, h, qk, ctx.sg_fn)
        return grad_qk, None
