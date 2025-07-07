import torch.nn as nn
from spikingjelly.activation_based import neuron


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

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode="m"
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode="m"
        )

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode="m"
        )

        self.attn_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode="m"
        )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode="m"
        )

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x_seq):
        T, N, C, H, W = x_seq.shape
        x_seq = x_seq.flatten(3)
        T, N, C, L = x_seq.shape
        x_for_qkv = x_seq.flatten(0, 1)  # [TN, C, L]

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, N, C, L).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(
            T, N, L, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, N, C, L).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(
            T, N, L, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, N, C, L).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(
            T, N, L, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()  # [T, N, H, L, CPH]

        x_seq = k.transpose(-2, -1) @ v
        x_seq = (q@x_seq) * self.scale
        x_seq = x_seq.transpose(3, 4).reshape(T, N, C, L).contiguous()
        x_seq = self.attn_lif(x_seq)

        x_seq = x_seq.flatten(0, 1)
        x_seq = self.proj_conv(x_seq)
        x_seq = self.proj_bn(x_seq).reshape(T, N, C, H, W).contiguous()
        x_seq = self.proj_lif(x_seq)

        return x_seq
