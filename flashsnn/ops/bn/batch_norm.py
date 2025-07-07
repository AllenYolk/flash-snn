"""Reference: attorch implementation of BN
https://github.com/BobMcDear/attorch/blob/main/attorch/batch_norm_kernels.py
where:
  x.shape = [N, C, L]
  block.shape = [ceil(N), 1, ceil(L)]; loop on L with chunk size BLOCK_L
Problem: locality?
Here, we adjust the blocking / chunking strategy:
  block.shape = [ceil(N), 1, BLOCK_L]; loop on N with chunk size 1
"""
from typing import Optional

import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.autograd as autograd

from flashsnn.utils import contiguous_and_device_guard, type_dict
from flashsnn.utils import amp_custom_fwd, amp_custom_bwd


def get_block_size(N, C, L):
    max_block_size = 16384 if C > 256 else 1024  # tuned by grid search
    BLOCK_L = triton.next_power_of_2(L)
    BLOCK_L = min(max_block_size, BLOCK_L)
    BLOCK_N = triton.cdiv(max_block_size, BLOCK_L)
    return BLOCK_N, BLOCK_L


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [2, 4, 8, 16]],
    key=[
        "BLOCK_N", "BLOCK_L", "affine", "track_running_stats", "is_train",
        "dtype", "running_stats_dtype"
    ],
    restore_value=[
        "output_ptr", "mean_ptr", "inv_std_ptr", "running_mean_ptr",
        "running_var_ptr"
    ]
)
@triton.jit
def batch_norm_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    inv_std_ptr,
    running_mean_ptr,
    running_var_ptr,
    momentum,
    eps,
    N: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    affine: tl.constexpr,
    save_stats: tl.constexpr,
    track_running_stats: tl.constexpr,
    is_train: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_L: tl.constexpr,  # i.e. chunk size
    dtype: tl.constexpr,
    running_stats_dtype: tl.constexpr,
):
    """Batch-normalizes the input, optionally adding a residual and fusing an 
    activation function.

    input.shape = [N, C, L]. For each program, block shape is [N, 1, L] (i.e. 
    one program for each channel). Each block is split into chunks with shape
    [BLOCK_N, 1, BLOCK_L], and a double-loop is used to calculate the stats.

    Notice that BN always use float32 for computation. We should cast the values
    to target dtypes before storing them. That's why we introduce `dtype` and
    `stats_dtype`. 
    """
    c = tl.program_id(axis=0)  # a.k.a. pid
    SN, SC, SL = C * L, L, 1  # stride
    NUMEL = N * L

    if is_train or not track_running_stats:  # use stats from the current batch
        count = 0
        mean = 0.0
        var = 0.0

        # if static_ranage on N, the compilation time cost is unaffordable!
        for n_start in tl.range(0, N, BLOCK_N):
            for l_start in tl.static_range(0, L, BLOCK_L):
                x_ptr = tl.make_block_ptr(
                    input_ptr,
                    shape=(N, C, L),
                    strides=(SN, SC, SL),
                    offsets=(n_start, c, l_start),
                    block_shape=(BLOCK_N, 1, BLOCK_L),
                    order=(2, 1, 0)
                )
                x = tl.load(x_ptr, boundary_check=(0, 2), padding_option="zero")
                x = x.to(tl.float32)

                cnt = min(BLOCK_L, L - l_start) * min(BLOCK_N, N - n_start)
                count += cnt

                prev_mean = mean
                mean += (tl.sum(x) - cnt*prev_mean) / count

                l_mask = (l_start + tl.arange(0, BLOCK_L)) < L
                deltas = tl.where(l_mask, (x-mean) * (x-prev_mean), 0.)
                var += tl.sum(deltas)

        var /= count
        inv_std = tl.rsqrt(var + eps)

        if save_stats:  # save for backward
            tl.store(mean_ptr + c, mean)
            tl.store(
                inv_std_ptr + c, inv_std
            )  # must be float32; no need for casting

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

    for n_start in tl.range(0, N, BLOCK_N):
        for l_start in tl.static_range(0, L, BLOCK_L):
            x_ptr = tl.make_block_ptr(
                input_ptr,
                shape=(N, C, L),
                strides=(SN, SC, SL),
                offsets=(n_start, c, l_start),
                block_shape=(BLOCK_N, 1, BLOCK_L),
                order=(2, 1, 0)
            )
            x = tl.load(x_ptr, boundary_check=(0, 2), padding_option="zero")
            x = x.to(tl.float32)

            y = (x-mean) * inv_std
            if affine:
                y = y*weight + bias

            y_ptr = tl.make_block_ptr(
                output_ptr,
                shape=(N, C, L),
                strides=(SN, SC, SL),
                offsets=(n_start, c, l_start),
                block_shape=(BLOCK_N, 1, BLOCK_L),
                order=(2, 1, 0)
            )
            tl.store(y_ptr, y.to(dtype), boundary_check=(0, 2))


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [2, 4, 8, 16]],
    key=["BLOCK_N", "BLOCK_L", "affine", "dtype", "grad_weight_dtype"],
    restore_value=["grad_input_ptr", "grad_weight_ptr", "grad_bias_ptr"]
)
@triton.jit
def batch_norm_backward_kernel(
    grad_output_ptr,
    input_ptr,
    mean_ptr,
    inv_std_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    affine: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_L: tl.constexpr,
    dtype: tl.constexpr,
    grad_weight_dtype: tl.constexpr,
):
    """The same blocking / chunking strategy as that of the forward kernel."""
    c = tl.program_id(axis=0)  # a.k.a. pid
    SN, SC, SL = C * L, L, 1  # stride
    NUMEL = N * L  # number of elements in the block

    mean = tl.load(mean_ptr + c)
    inv_std = tl.load(inv_std_ptr + c)  # scalars

    term1 = 0.0
    term2 = 0.0
    for n_start in tl.range(0, N, BLOCK_N):
        for l_start in tl.static_range(0, L, BLOCK_L):
            x_ptr = tl.make_block_ptr(
                input_ptr,
                shape=(N, C, L),
                strides=(SN, SC, SL),
                offsets=(n_start, c, l_start),
                block_shape=(BLOCK_N, 1, BLOCK_L),
                order=(2, 1, 0)
            )
            x = tl.load(x_ptr, boundary_check=(0, 2), padding_option="zero")
            x = x.to(tl.float32)
            grad_y_ptr = tl.make_block_ptr(
                grad_output_ptr,
                shape=(N, C, L),
                strides=(SN, SC, SL),
                offsets=(n_start, c, l_start),
                block_shape=(BLOCK_N, 1, BLOCK_L),
                order=(2, 1, 0)
            )
            grad_y = tl.load(
                grad_y_ptr, boundary_check=(0, 2), padding_option="zero"
            )
            grad_y = grad_y.to(tl.float32)

            y = (x-mean) * inv_std
            term1 += tl.sum(grad_y * y)
            term2 += tl.sum(grad_y)

    if affine:
        weight = tl.load(weight_ptr + c)
        weight_grad = 0.0
        bias_grad = 0.0
    else:
        weight = 1.

    term1 *= weight / NUMEL
    term2 *= weight / NUMEL

    for n_start in tl.range(0, N, BLOCK_N):
        for l_start in tl.static_range(0, L, BLOCK_L):
            x_ptr = tl.make_block_ptr(
                input_ptr,
                shape=(N, C, L),
                strides=(SN, SC, SL),
                offsets=(n_start, c, l_start),
                block_shape=(BLOCK_N, 1, BLOCK_L),
                order=(2, 1, 0)
            )
            x = tl.load(x_ptr, boundary_check=(0, 2), padding_option="zero")
            x = x.to(tl.float32)
            grad_y_ptr = tl.make_block_ptr(
                grad_output_ptr,
                shape=(N, C, L),
                strides=(SN, SC, SL),
                offsets=(n_start, c, l_start),
                block_shape=(BLOCK_N, 1, BLOCK_L),
                order=(2, 1, 0)
            )
            grad_y = tl.load(
                grad_y_ptr, boundary_check=(0, 2), padding_option="zero"
            )
            grad_y = grad_y.to(tl.float32)

            y = (x-mean) * inv_std
            grad_x = inv_std * (weight*grad_y - (term1*y + term2))

            grad_x_ptr = tl.make_block_ptr(
                grad_input_ptr,
                shape=(N, C, L),
                strides=(SN, SC, SL),
                offsets=(n_start, c, l_start),
                block_shape=(BLOCK_N, 1, BLOCK_L),
                order=(2, 1, 0)
            )
            tl.store(grad_x_ptr, grad_x.to(dtype), boundary_check=(0, 2))

            if affine:
                weight_grad += tl.sum(grad_y * y)
                bias_grad += tl.sum(grad_y)

    if affine:
        tl.store(grad_weight_ptr + c, weight_grad.to(grad_weight_dtype))
        tl.store(grad_bias_ptr + c, bias_grad.to(grad_weight_dtype))


class BatchNormFunction(autograd.Function):

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        training: bool,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        track_running_stats: bool = True,
    ) -> torch.Tensor:
        input_3d = input.unsqueeze(-1).reshape(*input.shape[:2], -1)

        affine = (weight is not None) and (bias is not None)
        requires_grad = (
            input.requires_grad or (affine and weight.requires_grad) or
            (affine and bias.requires_grad)
        )

        N, C, L = input_3d.shape
        output = torch.empty_like(input_3d)

        if requires_grad:  # mean and inv_std are always in float32
            mean = torch.empty(C, device=input.device, dtype=torch.float32)
            inv_std = torch.empty(C, device=input.device, dtype=torch.float32)
        else:
            mean = inv_std = None

        running_mean = input if (running_mean is None) else running_mean
        running_var = input if (running_var is None) else running_var

        BLOCK_N, BLOCK_L = get_block_size(N, C, L)

        batch_norm_forward_kernel[(C,)](
            input_3d,
            weight,
            bias,
            output,
            mean,
            inv_std,
            running_mean,
            running_var,
            momentum,
            eps,
            N,
            C,
            L,
            affine,
            save_stats=requires_grad,
            track_running_stats=track_running_stats,
            is_train=training,
            BLOCK_N=BLOCK_N,
            BLOCK_L=BLOCK_L,
            dtype=type_dict[output.dtype],
            running_stats_dtype=type_dict[running_mean.dtype]
        )

        ctx.affine = affine
        if requires_grad:
            ctx.save_for_backward(input, mean, inv_std, weight)
        return output.view_as(input)

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        input, mean, inv_std, weight = ctx.saved_tensors
        input_3d = input.unsqueeze(-1).reshape(*input.shape[:2], -1)
        grad_output = grad_output.view_as(input_3d)
        N, C, L = input_3d.shape
        grad_input = torch.empty_like(input_3d)

        if ctx.affine:
            grad_weight = torch.empty((C,), device=input.device)
            grad_bias = torch.empty_like(grad_weight)
        else:
            grad_weight = grad_bias = None

        BLOCK_N, BLOCK_L = get_block_size(N, C, L)

        batch_norm_backward_kernel[(C,)](
            grad_output,
            input_3d,
            mean,
            inv_std,
            weight,
            grad_input,
            grad_weight,
            grad_bias,
            N,
            C,
            L,
            ctx.affine,
            BLOCK_N,
            BLOCK_L,
            dtype=type_dict[grad_input.dtype],
            grad_weight_dtype=type_dict[grad_weight.dtype],
        )

        return (
            grad_input.view_as(input), None, grad_weight, grad_bias, None, None,
            None, None, None
        )


class BatchNorm1d(nn.BatchNorm1d):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device="cuda",
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device,
            dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        return BatchNormFunction.apply(
            input,
            self.training,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.momentum,
            self.eps,
            self.track_running_stats,
        )


class BatchNorm2d(nn.BatchNorm2d):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device="cuda",
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device,
            dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        return BatchNormFunction.apply(
            input,
            self.training,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.momentum,
            self.eps,
            self.track_running_stats,
        )
