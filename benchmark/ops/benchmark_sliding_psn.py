import sys

sys.path.append("./")

import torch
import triton
from spikingjelly.activation_based import neuron

from flashsnn.ops import psn

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


def sliding_psn_forward(x, weight, bias, T):
    weight = psn.GenerateSlidingPSNGemmWeightFunction.apply(weight, T)
    return psn.PSNFunction.apply(x, weight, bias.expand(T, 1))


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 17)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='neuron_type',
        # possible values for `line_arg``
        line_vals=['spikingjelly', 'spikingjelly-compile', 'triton'],
        # label name for the lines
        line_names=['SpikingJelly', 'SpikingJelly (compile)', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (NCL=8*700, k=10)",
        args={
            "k": 10,
            "NCL": 8 * 700
        },
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['NCL'],
        # different possible values for `x_name`
        x_vals=[128 * i for i in range(1, 51)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='neuron_type',
        # possible values for `line_arg``
        line_vals=['spikingjelly', 'spikingjelly-compile', 'triton'],
        # label name for the lines
        line_names=['SpikingJelly', 'SpikingJelly (compile)', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (T=16, k=10)",
        args={
            "k": 10,
            "T": 16
        },
    ),
])
def bacnmark(T, NCL, k, neuron_type):
    x = torch.randn([T, NCL], device=DEVICE, dtype=DTYPE)
    grad_y = torch.randn_like(x)
    x.requires_grad = True

    results = 0, 0, 0
    if neuron_type == "spikingjelly":
        f = neuron.SlidingPSN(k=k, step_mode="m").to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "spikingjelly-compile":
        f = neuron.SlidingPSN(k=k, step_mode="m").to(DEVICE)
        f = torch.compile(f, backend="inductor")
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "triton":
        weight = torch.randn([k], device=DEVICE, requires_grad=True)
        bias = torch.tensor(-1., device=DEVICE, requires_grad=True)
        f = psn.GenerateSlidingPSNGemmWeightFunction.apply
        results = triton.testing.do_bench(
            lambda: sliding_psn_forward(x, weight, bias, T).backward(grad_y),
            quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_sliding_psn",
        print_data=True,
        show_plots=True
    )
