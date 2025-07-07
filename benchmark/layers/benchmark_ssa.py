import sys

sys.path.append("./")

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
import triton

from flashsnn.layers import ssa

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


class OriginalSpikingSelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, backend="cupy"):
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
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

        self.attn_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

        self.proj_conv = nn.Conv1d(
            dim, dim, kernel_size=1, stride=1, bias=False
        )
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

    def forward(self, x_seq):
        functional.reset_net(self)
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
        ).permute(0, 1, 3, 2, 4).contiguous()  # [T, N, H, L, C_ph]

        x_seq = k.transpose(-2, -1) @ v
        x_seq = (q@x_seq) * self.scale
        x_seq = x_seq.transpose(3, 4).reshape(T, N, C, L).contiguous()
        x_seq = self.attn_lif(x_seq)

        x_seq = x_seq.flatten(0, 1)
        x_seq = self.proj_conv(x_seq)
        x_seq = self.proj_bn(x_seq).reshape(T, N, C, H, W).contiguous()
        x_seq = self.proj_lif(x_seq)

        return x_seq


class RefinedSpikingSelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, backend="cupy"):
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
        self.qkv_bn = nn.BatchNorm1d(dim * 3)
        self.qkv_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

        self.attn_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

        self.proj_conv = nn.Conv1d(
            dim, dim, kernel_size=1, stride=1, bias=False
        )
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend=backend, step_mode="m"
        )

    def forward(self, x_seq):
        functional.reset_net(self)
        T, N, C, L = x_seq.shape
        x_for_qkv = x_seq.flatten(0, 1)  # [TN, C, L]

        qkv = self.qkv_conv(x_for_qkv)  # [TN, C*3, L]
        qkv = self.qkv_bn(qkv).reshape(
            T, N, 3 * self.num_heads, C // self.num_heads, L
        ).contiguous()
        qkv = self.qkv_lif(qkv)
        q, k, v = qkv.chunk(3, dim=2)  # [T, N, num_heads, C_ph, L]

        x_seq = v @ k.transpose(-2, -1)
        x_seq = (x_seq@q) * self.scale  # [T, N, num_heads, C_ph, L]
        x_seq = self.attn_lif(x_seq).reshape(T, N, C, L)

        x_seq = x_seq.flatten(0, 1)  # [T*N, C, L]
        x_seq = self.proj_conv(x_seq)
        x_seq = self.proj_bn(x_seq).reshape(T, N, C, L)
        x_seq = self.proj_lif(x_seq)

        return x_seq


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[i for i in range(1, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'torch-refined', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Torch (refined)', 'Triton'],
        # line styles
        styles=[
            ('green', ':'),
            ('blue', '--'),
            ('red', '-.'),
            ('cyan', ':'),
            ('orange', '-'),
        ],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (N=16, C=256, L=14*14)",
        args={
            "N": 16,
            "C": 256,
            "L": 14
        },
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['C'],
        # different possible values for `x_name`
        x_vals=[64 * i for i in range(1, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'torch-refined', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Torch (refined)', 'Triton'],
        # line styles
        styles=[
            ('green', ':'),
            ('blue', '--'),
            ('red', '-.'),
            ('cyan', ':'),
            ('orange', '-'),
        ],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (T=4, N=16, L=14*14)",
        args={
            "T": 4,
            "N": 16,
            "L": 14
        },
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['L'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 11)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'torch-refined', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Torch (refined)', 'Triton'],
        # line styles
        styles=[
            ('green', ':'),
            ('blue', '--'),
            ('red', '-.'),
            ('cyan', ':'),
            ('orange', '-'),
        ],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (T=4, N=16, C=256)",
        args={
            "T": 4,
            "N": 16,
            "C": 256
        },
    ),
])
def bacnmark(T, N, C, L, implementation):
    results = 0, 0, 0

    if implementation == "torch":
        x = torch.randn([T, N, C, L, L], device=DEVICE, dtype=DTYPE)
        grad_y = torch.randn_like(x)
        x.requires_grad = True
        f = OriginalSpikingSelfAttention(dim=C).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif implementation == "torch-refined":
        x = torch.randn([T, N, C, L * L], device=DEVICE, dtype=DTYPE)
        grad_y = torch.randn_like(x)
        x.requires_grad = True
        f = RefinedSpikingSelfAttention(dim=C).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif implementation == "triton":
        x = torch.randn([T, N, C, L * L], device=DEVICE, dtype=DTYPE)
        grad_y = torch.randn_like(x)
        x.requires_grad = True
        f = ssa.SpikingSelfAttention(dim=C).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_ssa", print_data=True, show_plots=True
    )
