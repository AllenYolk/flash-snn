import triton
import triton.language as tl
import torch
from torch import autograd

from flashsnn.utils import contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd
from flashsnn.utils import get_device_capability


@triton.jit
def _gen_sliding_psn_gemm_weight_forward_kernel(
    input_weight_ptr,  # [k],
    weight_ptr,  # [T, T],
    T: tl.constexpr,
    k: tl.constexpr,
    BLOCK_T: tl.constexpr,  # >= T
):
    i = tl.program_id(0)
    end = i + 1
    start = i + 1 - k
    if start < 0:
        start = 0
    length = end - start
    if length > k:
        length = k

    offsets = tl.arange(0, BLOCK_T)
    mask = offsets < length

    input_start_ptr = input_weight_ptr + k - length
    input_vals = tl.load(input_start_ptr + offsets, mask=mask, other=0.)

    output_start_ptr = weight_ptr + i*T + start
    tl.store(output_start_ptr + offsets, input_vals, mask=mask)


@triton.jit
def _gen_sliding_psn_gemm_weight_backward_with_atomic_kernel(
    grad_weight_ptr,  # [T, T]
    grad_input_weight_ptr,  # [k]
    T: tl.constexpr,
    k: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    i = tl.program_id(0)

    end = i + 1
    start = i + 1 - k
    if start < 0:
        start = 0
    length = end - start
    if length > k:
        length = k

    offsets = tl.arange(0, BLOCK_T)
    mask = offsets < length

    grad_weight_ptr = grad_weight_ptr + i*T + start
    grad_weight = tl.load(grad_weight_ptr + offsets, mask=mask, other=0.)

    grad_input_weight_ptr = grad_input_weight_ptr + k - length
    tl.atomic_add(grad_input_weight_ptr + offsets, grad_weight, mask=mask)


@triton.jit
def _gen_sliding_psn_gemm_weight_backward_without_atomic_kernel(
    grad_weight_ptr,  # [T, T]
    grad_input_weight_ptr,  # [T, k]
    T: tl.constexpr,
    k: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    i = tl.program_id(0)

    end = i + 1
    start = i + 1 - k
    if start < 0:
        start = 0
    length = end - start
    if length > k:
        length = k

    offsets = tl.arange(0, BLOCK_T)
    mask = offsets < length

    grad_weight_ptr = grad_weight_ptr + i*T + start
    grad_weight = tl.load(grad_weight_ptr + offsets, mask=mask, other=0.)

    grad_input_weight_ptr = grad_input_weight_ptr + k - length + i*k
    tl.store(grad_input_weight_ptr + offsets, grad_weight, mask=mask)


def gen_sliding_psn_gemm_weight_forward(input_weight: torch.Tensor, T: int):
    weight = torch.zeros(
        [T, T],
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    k = input_weight.numel()
    BLOCK_T = triton.next_power_of_2(T)
    _gen_sliding_psn_gemm_weight_forward_kernel[(T,)](
        input_weight,
        weight,
        T=T,
        k=k,
        BLOCK_T=BLOCK_T,
    )
    return weight


def gen_sliding_psn_gemm_weight_backward_with_atomic(
    grad_weight: torch.Tensor, T: int, k: int
):
    grad_input_weight = torch.zeros(
        k, device=grad_weight.device, dtype=grad_weight.dtype
    )

    # Define block size
    BLOCK_T = triton.next_power_of_2(T)

    # Launch the kernel
    _gen_sliding_psn_gemm_weight_backward_with_atomic_kernel[(T,)](
        grad_weight,
        grad_input_weight,
        T=T,
        k=k,
        BLOCK_T=BLOCK_T,
    )
    return grad_input_weight


def gen_sliding_psn_gemm_weight_backward_without_atomic(
    grad_weight: torch.Tensor, T: int, k: int
):
    grad_input_weight = torch.zeros(
        [T, k],
        device=grad_weight.device,
        dtype=grad_weight.dtype,
    )

    # Define block size
    BLOCK_T = triton.next_power_of_2(T)

    # Launch the kernel
    _gen_sliding_psn_gemm_weight_backward_without_atomic_kernel[(T,)](
        grad_weight,
        grad_input_weight,
        T=T,
        k=k,
        BLOCK_T=BLOCK_T,
    )
    return grad_input_weight.sum(dim=0)


if get_device_capability()[0] < 7:
    gen_sliding_psn_gemm_weight_backward = gen_sliding_psn_gemm_weight_backward_without_atomic
else:
    gen_sliding_psn_gemm_weight_backward = gen_sliding_psn_gemm_weight_backward_with_atomic


class GenerateSlidingPSNGemmWeightFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(ctx, input_weight: torch.Tensor, T: int):
        weight = gen_sliding_psn_gemm_weight_forward(input_weight, T)
        ctx.T = T
        ctx.k = input_weight.numel()
        return weight

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_weight: torch.Tensor):
        T, k = ctx.T, ctx.k
        grad_input_weight = gen_sliding_psn_gemm_weight_backward(
            grad_weight, T, k
        )
        return grad_input_weight, None
