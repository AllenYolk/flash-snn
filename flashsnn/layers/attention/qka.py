import torch
import torch.nn as nn

from ..bn import BatchNorm1dLIF
from ..neurons.lif import LIF
from ...ops.qka import TokenQKAFunction, ChannelQKAFunction
from ...ops import surrogate_kernels

__all__ = ["QKAttention", "TokenQKAttention", "ChannelQKAttention"]


class QKAttention(nn.Module):

    def __init__(
        self, dim, num_heads=8, flash: bool = True, qka_type: str = "token"
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} should be divided by num_heads {num_heads}."
            )
        if qka_type not in ["token", "channel"]:
            raise ValueError(
                f"qka_type should be either 'token' or 'channel', "
                f"but got {qka_type}."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qka_type = qka_type

        self.qk_conv = nn.Conv1d(
            dim, dim * 2, kernel_size=1, stride=1, bias=False
        )
        self.qk_bn_lif = BatchNorm1dLIF(dim * 2, beta=0.5, detach_reset=True)

        if flash:
            if qka_type == "token":
                self.attn_kernel = TokenQKAFunction.apply
            else:
                self.attn_kernel = ChannelQKAFunction.apply
        else:
            self.attn_kernel = self._qka_forward_torch
            self.sum_dim = 3 if qka_type == "token" else 4
            self.attn_lif = LIF(beta=0.5, vth=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(
            dim, dim, kernel_size=1, stride=1, bias=False
        )
        self.proj_bn_lif = BatchNorm1dLIF(dim, beta=0.5, detach_reset=True)

    def _qka_forward_torch(self, qk, *args, **kwargs):
        # qk.shape = [T, N, 2, NUM_HEADS, Cph, L]
        # q, k = [T, N, NUM_HEADS, Cph, L]
        q, k = qk.flatten(2, 3).chunk(2, dim=2)
        q = torch.sum(q, dim=self.sum_dim, keepdim=True)
        # [T, N, NUM_HEADS, 1, L] if qka_type == "token"
        # [T, N, NUM_HEADS, Cph, 1] if qka_type == "channel"
        attn = self.attn_lif(q)
        x_seq = attn * k
        return x_seq  # [T, N, NUM_HEADS, Cph, L]

    def forward(self, x_seq):
        T, N, C, L = x_seq.shape

        qk = self.qk_conv(x_seq.flatten(0, 1)).reshape(T, N, 2 * C, L)
        qk = self.qk_bn_lif(qk)
        qk = qk.reshape(T, N, 2, self.num_heads, C // self.num_heads, L)

        x_seq = self.attn_kernel(qk, surrogate_kernels.atan_surrogate_backward)
        x_seq = x_seq.flatten(2, 3)  # [T, N, C, L]

        x_seq = self.proj_conv(x_seq.flatten(0, 1)).reshape(T, N, C, L)
        x_seq = self.proj_bn_lif(x_seq)

        return x_seq

    def extra_repr(self):
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, flash={self.flash}, "
            f"qka_type={self.qka_type}"
        )


class TokenQKAttention(QKAttention):

    def __init__(self, dim, num_heads=8, flash: bool = True):
        super().__init__(dim, num_heads, flash, qka_type="token")


class ChannelQKAttention(QKAttention):

    def __init__(self, dim, num_heads=8, flash: bool = True):
        super().__init__(dim, num_heads, flash, qka_type="channel")
