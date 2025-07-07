"""Reference: attorch implementation of BN
https://github.com/BobMcDear/attorch/blob/main/attorch/batch_norm_kernels.py
"""
from typing import Optional, Callable

import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.autograd as autograd

from flashsnn.ops.bn.batch_norm import batch_norm_backward_kernel
from flashsnn.ops import lif, surrogate_kernels
from flashsnn.utils import contiguous_and_device_guard, type_dict
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd


def get_block_size(T, N, C, L):
    max_block_size = 16384 if C > 256 else 1024  # tuned by grid search
    BLOCK_L = triton.next_power_of_2(L)
    BLOCK_L = min(max_block_size, BLOCK_L)
    BLOCK_N = triton.cdiv(max_block_size, BLOCK_L)
    return BLOCK_N, BLOCK_N, BLOCK_L


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [2, 4, 8, 16]],
    key=[
        "BLOCK_TN", "BLOCK_N", "BLOCK_L", "affine", "save_intermediates",
        "track_running_stats", "is_train", "residual", "soft_reset", "dtype",
        "running_stats_dtype"
    ],
    restore_value=[
        "s_seq_ptr", "h_seq_ptr", "mean_ptr", "inv_std_ptr", "running_mean_ptr",
        "running_var_ptr"
    ]
)
@triton.jit
def batch_norm_lif_forward_kernel(
    x_seq_ptr,
    weight_ptr,
    bias_ptr,
    r_seq_ptr,  # residual
    s_seq_ptr,  # w
    h_seq_ptr,  # w
    mean_ptr,  # w
    inv_std_ptr,  # w
    running_mean_ptr,  # rw
    running_var_ptr,  # rw
    momentum,
    eps,
    beta,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    affine: tl.constexpr,
    save_intermediates: tl.constexpr,
    track_running_stats: tl.constexpr,
    is_train: tl.constexpr,
    residual: tl.constexpr,
    soft_reset: tl.constexpr,
    BLOCK_TN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_L: tl.constexpr,  # i.e. chunk size
    dtype: tl.constexpr,
    running_stats_dtype: tl.constexpr,
):
    """Batch-normalizes the input and apply a LIF filter. Optionally add a 
    residual connection before the LIF layer.

    x_seq.shape = [T, N, C, L]. For each program, block shape is [T, N, 1, L] 
    (i.e. one program for each channel). 
    
    Stage 1. BN stats calculation. For each block, dimension T and N are first 
    merged. Then, the block is split into chunks with shape 
    [BLOCK_TN, 1, BLOCK_L], and a 2-loop (NT, L) is used to get the stats.

    Stage 2. BN and LIF forward. Each block is split into chunks with shape
    [1, BLOCK_N, 1, BLOCK_L], and a 3-loop (N, L, T) is used to calculate
    the output.

    Notice that BN always use float32 for computation. We should cast the values
    to target dtypes before storing them. That's why we introduce `dtype` and
    `stats_dtype`. 
    """
    c = tl.program_id(axis=0)  # a.k.a. pid
    TN = T * N
    ST, SN, SC, SL = N * C * L, C * L, L, 1  # stride
    NUMEL = TN * L

    # Stage 1. stats calculation
    if is_train or not track_running_stats:  # use stats from the current batch
        count = 0
        mean = 0.0
        var = 0.0

        # if static_ranage on TN, the compilation time cost is unaffordable!
        for tn_start in tl.range(0, TN, BLOCK_TN):
            for l_start in tl.static_range(0, L, BLOCK_L):
                x_ptr = tl.make_block_ptr(
                    x_seq_ptr,
                    shape=(TN, C, L),
                    strides=(SN, SC, SL),
                    offsets=(tn_start, c, l_start),
                    block_shape=(BLOCK_TN, 1, BLOCK_L),
                    order=(2, 1, 0)
                )
                x = tl.load(x_ptr, boundary_check=(0, 2), padding_option="zero")
                x = x.to(tl.float32)

                cnt = min(BLOCK_L, L - l_start) * min(BLOCK_TN, TN - tn_start)
                count += cnt

                prev_mean = mean
                mean += (tl.sum(x) - cnt*prev_mean) / count

                l_mask = (l_start + tl.arange(0, BLOCK_L)) < L
                deltas = tl.where(l_mask, (x-mean) * (x-prev_mean), 0.)
                var += tl.sum(deltas)

        var /= count
        inv_std = tl.rsqrt(var + eps)

        if save_intermediates:  # save for backward
            tl.store(mean_ptr + c, mean)
            tl.store(inv_std_ptr + c, inv_std)  # float32; casting not needed

        if track_running_stats:  # update stats
            running_mean_ptr += c
            running_var_ptr += c
            running_mean = tl.load(running_mean_ptr)
            running_var = tl.load(running_var_ptr)
            alpha = 1. - momentum
            running_mean = alpha*running_mean + momentum*mean
            running_var = alpha*running_var + momentum * var * NUMEL / (NUMEL-1)
            tl.store(running_mean_ptr, running_mean.to(running_stats_dtype))
            tl.store(running_var_ptr, running_var.to(running_stats_dtype))
    else:  # use running stats
        mean = tl.load(running_mean_ptr + c)
        inv_std = tl.rsqrt(tl.load(running_var_ptr + c) + eps)

    if affine:
        weight = tl.load(weight_ptr + c)  # scalar
        bias = tl.load(bias_ptr + c)

    # Stage 2. forward computation
    # mean, inv_std, weight, bias are all scalars!
    for n_start in tl.range(0, N, BLOCK_N):
        for l_start in tl.static_range(0, L, BLOCK_L):
            v = tl.zeros([1, BLOCK_N, 1, BLOCK_L], dtype=tl.float32)
            for t in tl.static_range(0, T, 1):
                x_ptr = tl.make_block_ptr(
                    x_seq_ptr,
                    shape=(T, N, C, L),
                    strides=(ST, SN, SC, SL),
                    offsets=(t, n_start, c, l_start),
                    block_shape=(1, BLOCK_N, 1, BLOCK_L),
                    order=(3, 2, 1, 0)
                )
                x = tl.load(x_ptr, boundary_check=(1, 3), padding_option="zero")
                x = x.to(tl.float32)

                y = (x-mean) * inv_std
                if affine:
                    y = y*weight + bias

                if residual:
                    r_ptr = tl.make_block_ptr(
                        r_seq_ptr,
                        shape=(T, N, C, L),
                        strides=(ST, SN, SC, SL),
                        offsets=(t, n_start, c, l_start),
                        block_shape=(1, BLOCK_N, 1, BLOCK_L),
                        order=(3, 2, 1, 0)
                    )
                    r = tl.load(
                        r_ptr, boundary_check=(1, 3), padding_option="zero"
                    )
                    y += r

                # LIF forward
                h = tl.fma(beta, v, y)
                s = (h >= 1.).to(tl.float32)
                if soft_reset:
                    v = h - s
                else:
                    v = h * (1.-s)

                s_ptr = tl.make_block_ptr(
                    s_seq_ptr,
                    shape=(T, N, C, L),
                    strides=(ST, SN, SC, SL),
                    offsets=(t, n_start, c, l_start),
                    block_shape=(1, BLOCK_N, 1, BLOCK_L),
                    order=(3, 2, 1, 0)
                )
                tl.store(s_ptr, s.to(dtype), boundary_check=(1, 3))
                if save_intermediates:
                    h_ptr = tl.make_block_ptr(
                        h_seq_ptr,
                        shape=(T, N, C, L),
                        strides=(ST, SN, SC, SL),
                        offsets=(t, n_start, c, l_start),
                        block_shape=(1, BLOCK_N, 1, BLOCK_L),
                        order=(3, 2, 1, 0)
                    )
                    tl.store(h_ptr, h, boundary_check=(1, 3))  # float32


#! `grad_lif_input` will be queried twice in the backward process. So, LIF's
#! backward is not included in the backward kernel (if fused, LIF's backward has
#! to be computed twice). Instead, we should compute LIF's backward in advance,
#! and then call BN's backward kernel! BN's backward kernel in batch_norm.py
#! is reused.


class BatchNormLIFFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx,
        x_seq: torch.Tensor,
        r_seq: Optional[torch.Tensor],  # residual
        training: bool,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        track_running_stats: bool = True,
        beta: float = 0.5,
        soft_reset: bool = False,
        detach_reset: bool = True,
        lif_bwd: Callable = lif.multistep_lif_hard_backward,
        sg_fn: Callable = surrogate_kernels.atan_surrogate_backward,
    ) -> torch.Tensor:
        x_seq_4d = x_seq.unsqueeze(-1).reshape(*x_seq.shape[:3], -1)

        residual = r_seq is not None
        affine = (weight is not None) and (bias is not None)
        requires_grad = (
            x_seq.requires_grad or (affine and weight.requires_grad) or
            (affine and bias.requires_grad) or
            (residual and r_seq.requires_grad)
        )

        T, N, C, L = x_seq_4d.shape
        s_seq = torch.empty_like(x_seq_4d)

        if requires_grad:  # mean and inv_std are always in float32
            mean = torch.empty(C, device=x_seq.device, dtype=torch.float32)
            inv_std = torch.empty(C, device=x_seq.device, dtype=torch.float32)
            h_seq = torch.empty_like(x_seq_4d, dtype=torch.float32)
            # mean, inv_std and h_seq will be saved for backward
        else:
            mean = inv_std = None

        running_mean = x_seq if (running_mean is None) else running_mean
        running_var = x_seq if (running_var is None) else running_var

        BLOCK_TN, BLOCK_N, BLOCK_L = get_block_size(T, N, C, L)

        batch_norm_lif_forward_kernel[(C,)](
            x_seq_4d,
            weight,
            bias,
            r_seq,
            s_seq,
            h_seq,
            mean,
            inv_std,
            running_mean,
            running_var,
            momentum,
            eps,
            beta,
            T,
            N,
            C,
            L,
            affine,
            save_intermediates=requires_grad,
            track_running_stats=track_running_stats,
            is_train=training,
            residual=residual,
            soft_reset=soft_reset,
            BLOCK_TN=BLOCK_TN,
            BLOCK_N=BLOCK_N,
            BLOCK_L=BLOCK_L,
            dtype=type_dict[s_seq.dtype],
            running_stats_dtype=type_dict[running_mean.dtype]
        )

        ctx.affine = affine
        ctx.residual = residual
        ctx.beta = beta
        ctx.soft_reset = soft_reset
        ctx.lif_bwd = lif_bwd
        ctx.sg_fn = sg_fn
        ctx.detach_reset = detach_reset
        if requires_grad:
            ex = [h_seq] if soft_reset else [h_seq, s_seq]
            ctx.save_for_backward(x_seq, mean, inv_std, weight, *ex)
        return s_seq.view_as(x_seq)

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_s_seq: torch.Tensor):
        x_seq, mean, inv_std, weight, *ex = ctx.saved_tensors
        x_seq_4d = x_seq.unsqueeze(-1).reshape(*x_seq.shape[:3], -1)

        grad_output = ctx.lif_bwd(
            grad_s_seq, *ex, ctx.beta, ctx.sg_fn, ctx.detach_reset, False
        )
        grad_output = grad_output.view_as(x_seq_4d)

        T, N, C, L = x_seq_4d.shape
        grad_input = torch.empty(T * N, C, L, device=x_seq.device)

        if ctx.affine:
            grad_weight = torch.empty((C,), device=x_seq.device)
            grad_bias = torch.empty_like(grad_weight)
        else:
            grad_weight = grad_bias = None

        _, BLOCK_N, BLOCK_L = get_block_size(T, N, C, L)

        grad_output_f = grad_output.flatten(0, 1)
        x_seq_4d_f = x_seq_4d.flatten(0, 1)
        batch_norm_backward_kernel[(C,)](
            grad_output_f,
            x_seq_4d_f,
            mean,
            inv_std,
            weight,
            grad_input,
            grad_weight,
            grad_bias,
            T * N,
            C,
            L,
            ctx.affine,
            BLOCK_N,
            BLOCK_L,
            dtype=type_dict[grad_input.dtype],
            grad_weight_dtype=type_dict[grad_weight.dtype],
        )

        return (
            grad_input.view_as(x_seq),
            grad_output.view_as(x_seq) if ctx.residual else None, None,
            grad_weight, grad_bias, None, None, None, None, None, None, None,
            None, None, None
        )


class BatchNorm1dLIF(nn.BatchNorm1d):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        beta: float = 0.5,
        soft_reset: bool = False,
        detach_reset: bool = True,
        sg_fn: Callable = surrogate_kernels.atan_surrogate_backward,
        device="cuda",
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device,
            dtype
        )
        self.beta = beta
        self.soft_reset = soft_reset
        self.detach_reset = detach_reset
        self.sg_fn = sg_fn

        if soft_reset:
            self.lif_bwd = lif.multistep_lif_soft_backward
        else:
            self.lif_bwd = lif.multistep_lif_hard_backward

    def forward(
        self,
        x_seq: torch.Tensor,
        r_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._check_input_dim(x_seq.flatten(0, 1))

        return BatchNormLIFFunction.apply(
            x_seq, r_seq, self.training, self.weight, self.bias,
            self.running_mean, self.running_var, self.momentum, self.eps,
            self.track_running_stats, self.beta, self.soft_reset,
            self.detach_reset, self.lif_bwd, self.sg_fn
        )


class BatchNorm2dLIF(nn.BatchNorm2d):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        beta: float = 0.5,
        soft_reset: bool = False,
        detach_reset: bool = True,
        sg_fn: Callable = surrogate_kernels.atan_surrogate_backward,
        device="cuda",
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device,
            dtype
        )
        self.beta = beta
        self.soft_reset = soft_reset
        self.detach_reset = detach_reset
        self.sg_fn = sg_fn

        if soft_reset:
            self.lif_bwd = lif.multistep_lif_soft_backward
        else:
            self.lif_bwd = lif.multistep_lif_hard_backward

    def forward(
        self,
        x_seq: torch.Tensor,
        r_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._check_input_dim(x_seq.flatten(0, 1))

        return BatchNormLIFFunction.apply(
            x_seq, r_seq, self.training, self.weight, self.bias,
            self.running_mean, self.running_var, self.momentum, self.eps,
            self.track_running_stats, self.beta, self.soft_reset,
            self.detach_reset, self.lif_bwd, self.sg_fn
        )
