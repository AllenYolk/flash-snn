import torch.nn as nn

from ..bn import BatchNorm1dLIF
from ..neurons import LIF


class SpikingSelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
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

        self.attn_lif = LIF(beta=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(
            dim, dim, kernel_size=1, stride=1, bias=False
        )
        self.proj_bn_lif = BatchNorm1dLIF(dim, beta=0.5, detach_reset=True)

    def forward(self, x_seq):
        T, N, C, L = x_seq.shape

        qkv = self.qkv_conv(x_seq.flatten(0, 1)).reshape(T, N, C * 3, L)
        qkv = self.qkv_bn_lif(qkv)
        qkv = qkv.reshape(T, N, self.num_heads * 3, C // self.num_heads, L)

        # TODO: wrap this in a Triton kernel
        q, k, v = qkv.chunk(3, dim=2)  # [T, N, num_heads, C_ph, L]
        # q, k, v are not contiguous (they are views of qkv).
        # aten.matmul will implicitly clone them to make them contiguous,
        # which is not efficient
        x_seq = v @ k.transpose(-2, -1)
        x_seq = (x_seq@q) * self.scale  # [T, N, num_heads, C_ph, L]

        x_seq = self.attn_lif(x_seq).reshape(T, N, C, L)

        x_seq = self.proj_conv(x_seq.flatten(0, 1)).reshape(T, N, C, L)
        x_seq = self.proj_bn_lif(x_seq)

        return x_seq
