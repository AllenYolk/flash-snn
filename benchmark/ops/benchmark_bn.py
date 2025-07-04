import sys

sys.path.append("./")

import torch
import torch.nn as nn
import triton

from attorch import nn as attnn
from flashsnn.ops import bn

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['TN'],
        # different possible values for `x_name`
        x_vals=[16 * i for i in range(1, 17)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'attorch', "triton"],
        # label name for the lines
        line_names=['Torch', 'attorch', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (C=256, L=64*64)",
        args={
            "C": 256,
            "L": 64 * 64
        },
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['C'],
        # different possible values for `x_name`
        x_vals=[64 * i for i in range(1, 17)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'attorch', 'triton'],
        # label name for the lines
        line_names=['Torch', 'attorch', "Triton"],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (TN=128, L=64*64)",
        args={
            "TN": 128,
            "L": 64 * 64
        },
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['L'],
        # different possible values for `x_name`
        x_vals=[16 * i * 16 * i for i in range(1, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'attorch', 'triton'],
        # label name for the lines
        line_names=['Torch', 'attorch', "Triton"],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (TN=128, C=256)",
        args={
            "TN": 128,
            "C": 256
        },
    ),
])
def bacnmark(TN, C, L, implementation):
    x = torch.randn([TN, C, L], device=DEVICE, dtype=DTYPE)
    grad_y = torch.randn_like(x)
    x.requires_grad = True

    results = 0, 0, 0
    if implementation == "torch":
        f = nn.BatchNorm1d(C).to(DEVICE)
    elif implementation == "attorch":
        f = attnn.BatchNorm1d(C).to(DEVICE)
    elif implementation == "triton":
        f = bn.BatchNorm1d(C).to(DEVICE)
    results = triton.testing.do_bench(
        lambda: f(x).backward(grad_y), quantiles=QUANTILES
    )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_bn", print_data=True, show_plots=True
    )
