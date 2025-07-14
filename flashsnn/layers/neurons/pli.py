import math

import torch
import torch.nn as nn

from flashsnn.ops import pli as pli_ops

__all__ = ["PLI"]


class PLI(nn.Module):

    def __init__(
        self,
        beta_init: float = 0.5,
        fwd_inplace: bool = False,
        bwd_inplace: bool = False
    ):
        super().__init__()
        _beta_init = self.sigmoid_reverse(beta_init)
        self._beta = nn.Parameter(torch.tensor(_beta_init), requires_grad=True)
        self.fwd_inplace = fwd_inplace
        self.bwd_inplace = bwd_inplace

    @staticmethod
    def sigmoid_reverse(y):
        return math.log(y / (1-y))

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    def forward(self, x_seq: torch.Tensor):
        return pli_ops.MultistepPLIFunction.apply(
            x_seq,
            self._beta.expand(x_seq.shape),  # apply sigmoid inside the kernel
            self.fwd_inplace,
            self.bwd_inplace,
        )

    def extra_repr(self):
        return f"beta={self.beta:.3f}, "
