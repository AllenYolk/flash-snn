from typing import Callable

import torch
import torch.nn as nn

from flashsnn.ops import surrogate_kernels, lif


class LIF(nn.Module):

    def __init__(
        self,
        beta: float = 0.5,
        soft_reset: bool = False,
        detach_reset: bool = True,
        sg_fn: Callable = surrogate_kernels.atan_surrogate_backward,
        fwd_inplace: bool = False,
        bwd_inplace: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.soft_reset = soft_reset
        self.detach_reset = detach_reset
        self.sg_fn = sg_fn
        self.fwd_inplace = fwd_inplace
        self.bwd_inplace = bwd_inplace

        if soft_reset:
            self.kernel = lif.MultistepLIFSoftFunction
        else:
            self.kernel = lif.MultistepLIFHardFunction

    def forward(self, x_seq: torch.Tensor):
        return self.kernel.apply(
            x_seq,
            self.beta,
            self.sg_fn,
            self.detach_reset,
            self.fwd_inplace,
            self.bwd_inplace,
        )
