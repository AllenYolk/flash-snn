import sys

sys.path.append("./")

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
import triton

from flashsnn.layers import qka

DEVICE = "cuda:4"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


class OriginalQKAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode='m'
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode='m'
        )

        self.attn_lif = neuron.LIFNode(
            tau=2.0,
            v_threshold=0.5,
            detach_reset=True,
            backend='cupy',
            step_mode='m'
        )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, backend='cupy', step_mode='m'
        )

    def forward(self, x):
        functional.reset_net(self)
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(
            T, B, self.num_heads, C // self.num_heads, N
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(
            T, B, self.num_heads, C // self.num_heads, N
        )

        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[i for i in range(1, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'triton', 'triton-flash'],
        # label name for the lines
        line_names=['Torch', 'Triton', 'Triton (flash)'],
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
        line_vals=['torch', 'triton', 'triton-flash'],
        # label name for the lines
        line_names=['Torch', 'Triton', 'Triton (flash)'],
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
        line_vals=['torch', 'triton', 'triton-flash'],
        # label name for the lines
        line_names=['Torch', 'Triton', 'Triton (flash)'],
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
        f = OriginalQKAttention(dim=C).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif implementation == "triton":
        x = torch.randn([T, N, C, L * L], device=DEVICE, dtype=DTYPE)
        grad_y = torch.randn_like(x)
        x.requires_grad = True
        f = qka.TokenQKAttention(dim=C, flash=False).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif implementation == "triton-flash":
        x = torch.randn([T, N, C, L * L], device=DEVICE, dtype=DTYPE)
        grad_y = torch.randn_like(x)
        x.requires_grad = True
        f = qka.TokenQKAttention(dim=C, flash=True).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_qka", print_data=True, show_plots=True
    )
