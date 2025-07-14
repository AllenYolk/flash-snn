import torch.nn as nn

from ..bn import BatchNorm1d, BatchNorm1dLIF
from ..neurons.lif import LIF
from ...ops.ssa import SSAFunction

__all__ = ["SpikingSelfAttention"]


class SpikingSelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, flash: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} should be divided by num_heads {num_heads}."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125

        self.qkv_conv = nn.Conv1d(
            dim, dim * 3, kernel_size=1, stride=1, bias=False
        )
        self.qkv_bn_lif = BatchNorm1dLIF(dim * 3, beta=0.5, detach_reset=True)

        if flash:
            self.attn_kernel = SSAFunction.apply
        else:
            self.attn_kernel = self._ssa_forward_torch
        self.attn_lif = LIF(beta=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(
            dim, dim, kernel_size=1, stride=1, bias=False
        )
        self.proj_bn_lif = BatchNorm1dLIF(dim, beta=0.5, detach_reset=True)

    @staticmethod
    def _ssa_forward_torch(qkv, scale):
        # qkv.shape = [T, N, 3, NUM_HEADS, Cph, L]
        # qt, kt, vt.shape = [T, N, NUM_HEADS, Cph, L]
        qt, kt, vt = qkv.flatten(2, 3).chunk(3, dim=2)
        x_seq = vt @ kt.transpose(-2, -1)
        x_seq = (x_seq@qt) * scale
        return x_seq

    def forward(self, x_seq):
        T, N, C, L = x_seq.shape

        qkv = self.qkv_conv(x_seq.flatten(0, 1)).reshape(T, N, 3 * C, L)
        qkv = self.qkv_bn_lif(qkv)
        qkv = qkv.reshape(T, N, 3, self.num_heads, C // self.num_heads, L)

        x_seq = self.attn_kernel(qkv, self.scale)
        x_seq = self.attn_lif(x_seq).reshape(T, N, C, L)

        x_seq = self.proj_conv(x_seq.flatten(0, 1)).reshape(T, N, C, L)
        x_seq = self.proj_bn_lif(x_seq)

        return x_seq

    def extra_repr(self):
        return f"dim={self.dim}, num_heads={self.num_heads}, flash={self.flash}"
