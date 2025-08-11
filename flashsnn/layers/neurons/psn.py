from typing import Callable
import math

import torch
import torch.nn as nn

from flashsnn.ops import surrogate_kernels
from flashsnn.ops import psn as psn_ops

__all__ = ["PSN", "SlidingPSN"]


class PSN(nn.Module):

    def __init__(
        self,
        T: int,
        sg_fn: Callable = surrogate_kernels.atan_surrogate_backward,
        inf_inplace: bool = False,
        bwd_inplace: bool = False,
    ):
        super().__init__()
        self.T = T
        self.sg_fn = sg_fn
        self.psn_weight = nn.Parameter(torch.empty([T, T]))
        self.psn_bias = nn.Parameter(torch.empty([T, 1]))
        nn.init.kaiming_uniform_(self.psn_weight, a=math.sqrt(5))
        nn.init.constant_(self.psn_bias, -1.)
        self.inf_inplace = inf_inplace
        self.bwd_inplace = bwd_inplace

    def forward(self, x_seq: torch.Tensor):
        return psn_ops.PSNFunction.apply(
            x_seq,
            self.psn_weight,
            self.psn_bias,
            self.sg_fn,
            self.inf_inplace,
            self.bwd_inplace,
        )

    def extra_repr(self):
        return f"T={self.T}, "


class SlidingPSN(nn.Module):

    def __init__(
        self,
        k: int,
        exp_init: bool = True,
        sg_fn: Callable = surrogate_kernels.atan_surrogate_backward,
        inf_inplace: bool = False,
        bwd_inplace: bool = False,
    ):
        super().__init__()
        self.k = k
        self.sg_fn = sg_fn

        if exp_init:
            weight = torch.ones([k])
            for i in range(k - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.
        else:
            weight = torch.ones([1, k])
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight = weight[0]
        self.spsn_weight = nn.Parameter(weight)
        self.spsn_bias = nn.Parameter(torch.tensor(-1.))
        self.inf_inplace = inf_inplace
        self.bwd_inplace = bwd_inplace

    def forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        weight = psn_ops.GenerateSlidingPSNGemmWeightFunction.apply(
            self.spsn_weight, T
        )
        return psn_ops.PSNFunction.apply(
            x_seq,
            weight,
            self.spsn_bias.expand(T, 1),
            self.sg_fn,
            self.inf_inplace,
            self.bwd_inplace,
        )
