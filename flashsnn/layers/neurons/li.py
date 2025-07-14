import torch
import torch.nn as nn

from flashsnn.ops import li as li_ops

__all__ = ["LI"]


class LI(nn.Module):

    def __init__(
        self,
        beta: float = 0.5,
        fwd_inplace: bool = False,
        bwd_inplace: bool = False
    ):
        super().__init__()
        self.beta = beta
        self.fwd_inplace = fwd_inplace
        self.bwd_inplace = bwd_inplace

    def forward(self, x_seq: torch.Tensor):
        return li_ops.MultistepLIFunction.apply(
            x_seq,
            self.beta,
            self.fwd_inplace,
            self.bwd_inplace,
        )

    def extra_repr(self):
        return f"beta={self.beta:.3f}, "
