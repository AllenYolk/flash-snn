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
        triton.Config({
            "BLOCK_LQ": b,
            "BLOCK_LK": b
        }, num_warps=w, num_stages=1)
        for b in [16, 32, 64]
        for w in [8, 16, 32]
    ],
    key=["TN", "NUM_HEADS", "Cph", "L", "BLOCK_Cph", "dtype"],
    restore_value=["output_ptr"],
)
@triton.jit
def _ssa_forward_kernel(
    qkv_ptr,  # [TN, 3, NUM_HEADS, Cph, L]
    output_ptr,  # [TN, NUM_HEADS, Cph, L]; w
    scale,
    TN: tl.constexpr,  # for autotune
    NUM_HEADS: tl.constexpr,
    Cph: tl.constexpr,
    L: tl.constexpr,
    BLOCK_LQ: tl.constexpr,
    BLOCK_LK: tl.constexpr,
    BLOCK_Cph: tl.constexpr,  # >= Cph
    dtype: tl.constexpr,
):
    tn = tl.program_id(0)
    head = tl.program_id(1)
    lq_start = tl.program_id(2) * BLOCK_LQ

    # locate the [Cph, L] matrices
    S_head = Cph * L
    S_qkv = NUM_HEADS * S_head
    S_tn = 3 * S_qkv
    q_base = qkv_ptr + tn*S_tn + head*S_head
    k_base = q_base + S_qkv
    v_base = k_base + S_qkv

    q_ptrs = tl.make_block_ptr(
        q_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lq_start, 0),
        block_shape=(BLOCK_LQ, BLOCK_Cph),
        order=(1, 0)
    )  # transpose loading!
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    acc = tl.zeros((BLOCK_LQ, BLOCK_Cph), dtype=dtype)
    scale = tl.full((1,), scale, dtype=dtype)

    for lk_start in tl.static_range(0, L, BLOCK_LK):
        k_ptrs = tl.make_block_ptr(
            k_base,
            shape=(Cph, L),
            strides=(L, 1),
            offsets=(0, lk_start),
            block_shape=(BLOCK_Cph, BLOCK_LK),
            order=(1, 0)
        )
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        v_ptrs = tl.make_block_ptr(
            v_base,
            shape=(L, Cph),
            strides=(1, L),
            offsets=(lk_start, 0),
            block_shape=(BLOCK_LK, BLOCK_Cph),
            order=(1, 0)
        )
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        if BLOCK_Cph <= BLOCK_LQ and BLOCK_Cph <= BLOCK_LK:
            kv = tl.dot(
                k,
                v,
                input_precision="ieee",
                out_dtype=dtype,
            )
            acc = tl.dot(
                q,
                kv,
                acc=acc,
                input_precision="ieee",
                out_dtype=dtype,
            )
        else:
            qk = tl.dot(
                q,
                k,
                input_precision="ieee",
                out_dtype=dtype,
            )
            acc = tl.dot(
                qk,
                v,
                acc=acc,
                input_precision="ieee",
                out_dtype=dtype,
            )

    acc = acc * scale

    output_base = output_ptr + tn*S_qkv + head*S_head
    output_ptrs = tl.make_block_ptr(
        output_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lq_start, 0),
        block_shape=(BLOCK_LQ, BLOCK_Cph),
        order=(1, 0)
    )
    tl.store(output_ptrs, acc, boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_LQ": b,
            "BLOCK_LK": b
        }, num_warps=w, num_stages=1)
        for b in [16, 32, 64]
        for w in [8, 16, 32]
    ],
    key=["TN", "NUM_HEADS", "Cph", "L", "BLOCK_Cph", "dtype"],
    restore_value=["grad_qkv_ptr"],
)
@triton.jit
def _ssa_backward_kernel_with_atomic(
    grad_output_ptr,  # [TN, NUM_HEADS, Cph, L]
    qkv_ptr,  # [TN, 3, NUM_HEADS, Cph, L]
    grad_qkv_ptr,  # [TN, 3, NUM_HEADS, Cph, L]; w
    scale,
    TN: tl.constexpr,  # for autotune
    NUM_HEADS: tl.constexpr,
    Cph: tl.constexpr,
    L: tl.constexpr,
    BLOCK_LQ: tl.constexpr,
    BLOCK_LK: tl.constexpr,
    BLOCK_Cph: tl.constexpr,  # >= Cph
    dtype: tl.constexpr,
):
    tn = tl.program_id(0)
    head = tl.program_id(1)
    lk_start = tl.program_id(2) * BLOCK_LK

    # locate the [Cph, L] matrices
    S_head = Cph * L
    S_qkv = NUM_HEADS * S_head
    S_tn = 3 * S_qkv
    q_base = qkv_ptr + tn*S_tn + head*S_head
    k_base = q_base + S_qkv
    v_base = k_base + S_qkv
    grad_q_base = grad_qkv_ptr + tn*S_tn + head*S_head
    grad_k_base = grad_q_base + S_qkv
    grad_v_base = grad_k_base + S_qkv
    grad_output_base = grad_output_ptr + tn*S_qkv + head*S_head

    cphs = tl.arange(0, BLOCK_Cph)
    lqs = tl.arange(0, BLOCK_LQ)

    k_ptrs = tl.make_block_ptr(
        k_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lk_start, 0),
        block_shape=(BLOCK_LK, BLOCK_Cph),
        order=(1, 0)
    )
    kt = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v_ptrs = tl.make_block_ptr(
        v_base,
        shape=(Cph, L),
        strides=(L, 1),
        offsets=(0, lk_start),
        block_shape=(BLOCK_Cph, BLOCK_LK),
        order=(1, 0)
    )
    vt = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

    grad_k = tl.zeros((BLOCK_Cph, BLOCK_LK), dtype=dtype)
    grad_v = tl.zeros((BLOCK_LK, BLOCK_Cph), dtype=dtype)
    scale = tl.full((1,), scale, dtype=dtype)

    for lq_start in tl.static_range(0, L, BLOCK_LQ):
        q_ptrs = tl.make_block_ptr(
            q_base,
            shape=(Cph, L),
            strides=(L, 1),
            offsets=(0, lq_start),
            block_shape=(BLOCK_Cph, BLOCK_LQ),
            order=(1, 0)
        )  # the transpose of the q in the forward kernel
        qt = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        grad_output_ptrs = tl.make_block_ptr(
            grad_output_base,
            shape=(L, Cph),
            strides=(1, L),
            offsets=(lq_start, 0),
            block_shape=(BLOCK_LQ, BLOCK_Cph),
            order=(1, 0)
        )
        grad_output = tl.load(
            grad_output_ptrs, boundary_check=(0, 1), padding_option="zero"
        )

        if BLOCK_Cph <= BLOCK_LQ and BLOCK_Cph <= BLOCK_LK:
            # grad_q = scale * grad_output * (v^t * k^t)
            vtkt = tl.dot(
                vt,
                kt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_q = tl.dot(
                grad_output,
                vtkt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            # grad_k = scale * (q^T * grad_output) * v^T
            qt_grad_output = tl.dot(
                qt,
                grad_output,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_k = tl.dot(
                qt_grad_output,
                vt,
                acc=grad_k,
                input_precision="ieee",
                out_dtype=dtype,
            )
            # grad_v = scale * k^T * (q^T * grad_output)
            grad_v = tl.dot(
                kt,
                qt_grad_output,
                acc=grad_v,
                input_precision="ieee",
                out_dtype=dtype,
            )
        else:
            # grad_q = scale * (grad_output * v^t) * k^t
            grad_output_vt = tl.dot(
                grad_output,
                vt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_q = tl.dot(
                grad_output_vt,
                kt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            # grad_k = scale * q^T * (grad_output * v^T)
            grad_k = tl.dot(
                qt,
                grad_output_vt,
                acc=grad_k,
                input_precision="ieee",
                out_dtype=dtype,
            )
            # grad_v = scale * (k^T * q^T) * grad_output
            ktqt = tl.dot(
                kt,
                qt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_v = tl.dot(
                ktqt,
                grad_output,
                acc=grad_v,
                input_precision="ieee",
                out_dtype=dtype,
            )

        qrow = lqs[:, None] + lq_start
        qcol = cphs[None, :]
        grad_q_ptrs = grad_q_base + qrow + qcol*L
        qmask = (qrow < L) & (qcol < Cph)
        tl.atomic_add(grad_q_ptrs, grad_q * scale, mask=qmask)

    grad_k_ptrs = tl.make_block_ptr(
        grad_k_base,
        shape=(Cph, L),
        strides=(L, 1),
        offsets=(0, lk_start),
        block_shape=(BLOCK_Cph, BLOCK_LK),
        order=(1, 0)
    )
    tl.store(grad_k_ptrs, grad_k * scale, boundary_check=(0, 1))
    grad_v_ptrs = tl.make_block_ptr(
        grad_v_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lk_start, 0),
        block_shape=(BLOCK_LK, BLOCK_Cph),
        order=(1, 0)
    )
    tl.store(grad_v_ptrs, grad_v * scale, boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_LQ": b,
            "BLOCK_LK": b
        }, num_warps=w, num_stages=1)
        for b in [16, 32, 64]
        for w in [8, 16, 32]
    ],
    key=["TN", "NUM_HEADS", "Cph", "L", "BLOCK_Cph", "dtype"],
    restore_value=["grad_qkv_ptr"],
)
@triton.jit
def _ssa_backward_kernel_without_atomic(
    grad_output_ptr,  # [TN, NUM_HEADS, Cph, L]
    qkv_ptr,  # [TN, 3, NUM_HEADS, Cph, L]
    grad_qkv_ptr,  # [TN, 3, NUM_HEADS, Cph, L]; w
    scale,
    TN: tl.constexpr,  # for autotune
    NUM_HEADS: tl.constexpr,
    Cph: tl.constexpr,
    L: tl.constexpr,
    BLOCK_LQ: tl.constexpr,
    BLOCK_LK: tl.constexpr,
    BLOCK_Cph: tl.constexpr,  # >= Cph
    dtype: tl.constexpr,
):
    tnh = tl.program_id(0)
    tn = tnh // NUM_HEADS
    head = tnh % NUM_HEADS

    # locate the [Cph, L] matrices
    S_head = Cph * L
    S_qkv = NUM_HEADS * S_head
    S_tn = 3 * S_qkv
    q_base = qkv_ptr + tn*S_tn + head*S_head
    k_base = q_base + S_qkv
    v_base = k_base + S_qkv
    grad_q_base = grad_qkv_ptr + tn*S_tn + head*S_head
    grad_k_base = grad_q_base + S_qkv
    grad_v_base = grad_k_base + S_qkv
    grad_output_base = grad_output_ptr + tn*S_qkv + head*S_head

    scale = tl.full((1,), scale, dtype=dtype)

    # Stage 1. Calculate grad_q
    lq_start = tl.program_id(1) * BLOCK_LQ
    q_ptrs = tl.make_block_ptr(
        q_base,
        shape=(Cph, L),
        strides=(L, 1),
        offsets=(0, lq_start),
        block_shape=(BLOCK_Cph, BLOCK_LQ),
        order=(1, 0)
    )  # the transpose of the q in the forward kernel
    qt = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    grad_output_ptrs = tl.make_block_ptr(
        grad_output_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lq_start, 0),
        block_shape=(BLOCK_LQ, BLOCK_Cph),
        order=(1, 0)
    )
    grad_output = tl.load(
        grad_output_ptrs, boundary_check=(0, 1), padding_option="zero"
    )

    grad_q = tl.zeros((BLOCK_LQ, BLOCK_Cph), dtype=dtype)

    for lk_start in tl.static_range(0, L, BLOCK_LK):
        k_ptrs = tl.make_block_ptr(
            k_base,
            shape=(L, Cph),
            strides=(1, L),
            offsets=(lk_start, 0),
            block_shape=(BLOCK_LK, BLOCK_Cph),
            order=(1, 0)
        )
        kt = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        v_ptrs = tl.make_block_ptr(
            v_base,
            shape=(Cph, L),
            strides=(L, 1),
            offsets=(0, lk_start),
            block_shape=(BLOCK_Cph, BLOCK_LK),
            order=(1, 0)
        )
        vt = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

        if BLOCK_Cph <= BLOCK_LQ and BLOCK_Cph <= BLOCK_LK:
            # grad_q = scale * grad_output * (v^t * k^t)
            vtkt = tl.dot(
                vt,
                kt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_q = tl.dot(
                grad_output,
                vtkt,
                acc=grad_q,
                input_precision="ieee",
                out_dtype=dtype
            )
        else:
            # grad_q = scale * (grad_output * v^t) * k^t
            grad_output_vt = tl.dot(
                grad_output,
                vt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_q = tl.dot(
                grad_output_vt,
                kt,
                acc=grad_q,
                input_precision="ieee",
                out_dtype=dtype
            )

    grad_q_ptrs = tl.make_block_ptr(
        grad_q_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lq_start, 0),
        block_shape=(BLOCK_LQ, BLOCK_Cph),
        order=(1, 0)
    )
    tl.store(grad_q_ptrs, grad_q * scale, boundary_check=(0, 1))

    # Stage 2. Calculate grad_k and grad_v
    lk_start = tl.program_id(2) * BLOCK_LK
    k_ptrs = tl.make_block_ptr(
        k_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lk_start, 0),
        block_shape=(BLOCK_LK, BLOCK_Cph),
        order=(1, 0)
    )
    kt = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v_ptrs = tl.make_block_ptr(
        v_base,
        shape=(Cph, L),
        strides=(L, 1),
        offsets=(0, lk_start),
        block_shape=(BLOCK_Cph, BLOCK_LK),
        order=(1, 0)
    )
    vt = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

    grad_k = tl.zeros((BLOCK_Cph, BLOCK_LK), dtype=dtype)
    grad_v = tl.zeros((BLOCK_LK, BLOCK_Cph), dtype=dtype)

    for lq_start in tl.static_range(0, L, BLOCK_LQ):
        q_ptrs = tl.make_block_ptr(
            q_base,
            shape=(Cph, L),
            strides=(L, 1),
            offsets=(0, lq_start),
            block_shape=(BLOCK_Cph, BLOCK_LQ),
            order=(1, 0)
        )  # the transpose of the q in the forward kernel
        qt = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        grad_output_ptrs = tl.make_block_ptr(
            grad_output_base,
            shape=(L, Cph),
            strides=(1, L),
            offsets=(lq_start, 0),
            block_shape=(BLOCK_LQ, BLOCK_Cph),
            order=(1, 0)
        )
        grad_output = tl.load(
            grad_output_ptrs, boundary_check=(0, 1), padding_option="zero"
        )

        if BLOCK_Cph <= BLOCK_LQ and BLOCK_Cph <= BLOCK_LK:
            # grad_k = scale * (q^T * grad_output) * v^T
            qt_grad_output = tl.dot(
                qt,
                grad_output,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_k = tl.dot(
                qt_grad_output,
                vt,
                acc=grad_k,
                input_precision="ieee",
                out_dtype=dtype,
            )
            # grad_v = scale * k^T * (q^T * grad_output)
            grad_v = tl.dot(
                kt,
                qt_grad_output,
                acc=grad_v,
                input_precision="ieee",
                out_dtype=dtype,
            )
        else:
            # grad_k = scale * q^T * (grad_output * v^T)
            grad_output_vt = tl.dot(
                grad_output,
                vt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_k = tl.dot(
                qt,
                grad_output_vt,
                acc=grad_k,
                input_precision="ieee",
                out_dtype=dtype,
            )
            # grad_v = scale * (k^T * q^T) * grad_output
            ktqt = tl.dot(
                kt,
                qt,
                input_precision="ieee",
                out_dtype=dtype,
            )
            grad_v = tl.dot(
                ktqt,
                grad_output,
                acc=grad_v,
                input_precision="ieee",
                out_dtype=dtype,
            )

    grad_k_ptrs = tl.make_block_ptr(
        grad_k_base,
        shape=(Cph, L),
        strides=(L, 1),
        offsets=(0, lk_start),
        block_shape=(BLOCK_Cph, BLOCK_LK),
        order=(1, 0)
    )
    tl.store(grad_k_ptrs, grad_k * scale, boundary_check=(0, 1))
    grad_v_ptrs = tl.make_block_ptr(
        grad_v_base,
        shape=(L, Cph),
        strides=(1, L),
        offsets=(lk_start, 0),
        block_shape=(BLOCK_LK, BLOCK_Cph),
        order=(1, 0)
    )
    tl.store(grad_v_ptrs, grad_v * scale, boundary_check=(0, 1))


def ssa_forward(qkv, scale):
    # qkv.shape = [T, N, 3, NUM_HEADS, Cph, L]
    TN = qkv.shape[0] * qkv.shape[1]
    NUM_HEADS = qkv.shape[3]
    Cph = qkv.shape[4]
    L = qkv.shape[5]

    BLOCK_Cph = max(triton.next_power_of_2(Cph), 16)

    output = torch.empty_like(qkv[:, :, 0])
    grid = lambda meta: (TN, NUM_HEADS, triton.cdiv(L, meta['BLOCK_LQ']))
    _ssa_forward_kernel[grid](
        qkv,
        output,
        scale,
        TN,
        NUM_HEADS,
        Cph,
        L,
        BLOCK_Cph=BLOCK_Cph,
        dtype=type_dict[qkv.dtype]
    )

    return output


def ssa_backward_with_atomic(grad_output, qkv, scale):
    # grad_output.shape = [T, N, NUM_HEADS, Cph, L]
    # qkv.shape = [T, N, 3, NUM_HEADS, Cph, L]
    TN = qkv.shape[0] * qkv.shape[1]
    NUM_HEADS = qkv.shape[3]
    Cph = qkv.shape[4]
    L = qkv.shape[5]

    BLOCK_Cph = max(triton.next_power_of_2(Cph), 16)

    #! atomic-added buffers must be init to 0, not empty()
    grad_qkv = torch.zeros_like(qkv)
    grid = lambda meta: (TN, NUM_HEADS, triton.cdiv(L, meta['BLOCK_LK']))
    _ssa_backward_kernel_with_atomic[grid](
        grad_output,
        qkv,
        grad_qkv,
        scale,
        TN,
        NUM_HEADS,
        Cph,
        L,
        BLOCK_Cph=BLOCK_Cph,
        dtype=type_dict[grad_output.dtype]
    )

    return grad_qkv


def ssa_backward_without_atomic(grad_output, qkv, scale):
    # grad_output.shape = [T, N, NUM_HEADS, Cph, L]
    # qkv.shape = [T, N, 3, NUM_HEADS, Cph, L]
    TN = qkv.shape[0] * qkv.shape[1]
    NUM_HEADS = qkv.shape[3]
    Cph = qkv.shape[4]
    L = qkv.shape[5]

    BLOCK_Cph = max(triton.next_power_of_2(Cph), 16)

    grad_qkv = torch.empty_like(qkv)
    grid = lambda meta: (
        TN * NUM_HEADS,
        triton.cdiv(L, meta['BLOCK_LQ']),
        triton.cdiv(L, meta['BLOCK_LK']),
    )
    _ssa_backward_kernel_without_atomic[grid](
        grad_output,
        qkv,
        grad_qkv,
        scale,
        TN,
        NUM_HEADS,
        Cph,
        L,
        BLOCK_Cph=BLOCK_Cph,
        dtype=type_dict[grad_output.dtype]
    )

    return grad_qkv


if get_device_capability()[0] < 7:
    ssa_backward = ssa_backward_without_atomic
else:
    ssa_backward = ssa_backward_with_atomic


class SSAFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, qkv: torch.Tensor, scale: float):
        o = ssa_forward(qkv, scale)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(qkv)
            ctx.scale = scale
        return o

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        qkv, = ctx.saved_tensors
        grad_qkv = ssa_backward(grad_output, qkv, ctx.scale)
        return grad_qkv, None
