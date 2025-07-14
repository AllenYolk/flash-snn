from typing import Callable
import math

import torch
import torch.nn as nn

from flashsnn.ops import surrogate_kernels
from flashsnn.ops import plif as plif_ops

__all__ = ["PLIF"]


class PLIF(nn.Module):

    def __init__(
        self,
        beta_init: float = 0.5,
        soft_reset: bool = False,
        detach_reset: bool = True,
        sg_fn: Callable = surrogate_kernels.atan_surrogate_backward,
        fwd_inplace: bool = False,
        bwd_inplace: bool = False,
    ):
        super().__init__()
        _beta_init = self.sigmoid_reverse(beta_init)
        self._beta = nn.Parameter(torch.tensor(_beta_init), requires_grad=True)
        self.soft_reset = soft_reset
        self.detach_reset = detach_reset
        self.sg_fn = sg_fn
        self.fwd_inplace = fwd_inplace
        self.bwd_inplace = bwd_inplace

        if soft_reset:
            self.kernel = plif_ops.MultistepPLIFSoftFunction
        else:
            self.kernel = plif_ops.MultistepPLIFHardFunction

    @staticmethod
    def sigmoid_reverse(y):
        return math.log(y / (1-y))

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    def forward(self, x_seq: torch.Tensor):
        return self.kernel.apply(
            x_seq,
            self._beta.expand(x_seq.shape),  # apply sigmoid inside the kernel
            self.sg_fn,
            self.detach_reset,
            self.fwd_inplace,
            self.bwd_inplace,
        )

    def extra_repr(self):
        return (
            f"beta={self.beta:.3f}, "
            f"soft_reset={self.soft_reset}, "
            f"detach_reset={self.detach_reset}, "
        )
