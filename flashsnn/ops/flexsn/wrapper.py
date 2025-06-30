from functools import lru_cache
from typing import List

import torch
from torch import autograd
import triton

from flashsnn.utils import type_dict, contiguous_and_device_guard
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd
from flashsnn.utils import get_multiprocessor_count


@lru_cache(maxsize=None)
def _get_block_size(NCL, device_idx):
    BLOCK_NCL = triton.next_power_of_2(
        triton.cdiv(NCL, get_multiprocessor_count(device_idx))
    )
    BLOCK_NCL = min(1024, max(128, BLOCK_NCL))
    return BLOCK_NCL


def flexsn_inference(x_seq: torch.Tensor, f: triton.JITFunction):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, x_seq.device.index)
    s_seq = torch.empty_like(x_seq)
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    f[grid](
        x_seq,
        s_seq,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return s_seq


def flexsn_forward(x_seq: torch.Tensor, f: triton.JITFunction, n_returns: int):
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, x_seq.device.index)
    returns = [torch.empty_like(x_seq) for i in range(n_returns)]
    dtype = x_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    f[grid](
        x_seq,
        *returns,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return tuple(returns)


def flexsn_backward(
    grad_s_seq: torch.Tensor,
    required_results: List[torch.Tensor],
    f: triton.JITFunction,
):
    T = grad_s_seq.shape[0]
    NCL = grad_s_seq[0].numel()
    BLOCK_NCL = _get_block_size(NCL, grad_s_seq.device.index)
    grad_x_seq = torch.empty_like(grad_s_seq)
    dtype = grad_s_seq.dtype
    grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

    f[grid](
        grad_s_seq,
        *required_results,
        grad_x_seq,
        T=T,
        NCL=NCL,
        BLOCK_NCL=BLOCK_NCL,
        dtype=type_dict[dtype],
    )
    return grad_x_seq


class FlexSNFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx,
        x_seq: torch.Tensor,
        info: dict,
        fn_inf: triton.JITFunction,
        fn_fwd: triton.JITFunction,
        fn_bwd: triton.JITFunction,
    ):
        if any(ctx.needs_input_grad):
            results = flexsn_forward(
                x_seq, fn_fwd, info["N_fwd_kernel_returns"]
            )
            s_seq = results[0]
            to_save = []
            for i in info["extra_return_mapping"]:
                to_save.append(results[i])
            ctx.save_for_backward(*to_save)
            ctx.fn_bwd = fn_bwd
        else:
            s_seq = flexsn_inference(x_seq, fn_inf)
        return s_seq

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        required_results = ctx.saved_tensors
        fn_bwd = ctx.fn_bwd
        grad_x_seq = flexsn_backward(grad_s_seq, required_results, fn_bwd)
        return grad_x_seq, None, None, None, None
