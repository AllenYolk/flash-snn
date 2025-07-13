from typing import Callable, Optional

import torch
import torch.nn as nn

from flashsnn.ops.bn import BatchNormFunction, BatchNormLIFFunction
from flashsnn.ops import lif as lif_ops
from flashsnn.ops import surrogate_kernels


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
            self.lif_bwd = lif_ops.multistep_lif_soft_backward
        else:
            self.lif_bwd = lif_ops.multistep_lif_hard_backward

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
            self.lif_bwd = lif_ops.multistep_lif_soft_backward
        else:
            self.lif_bwd = lif_ops.multistep_lif_hard_backward

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
