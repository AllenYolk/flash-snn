import math
import sys

sys.path.append("./")

import torch
import torch.nn as nn
import triton
from spikingjelly.activation_based import surrogate

from flashsnn.ops import psn

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]
SG = "atan"


def get_psn_autograd_function(sg):
    if sg.lower() == "atan":
        s1 = "Atan"
    else:
        s1 = "Atan"
    return getattr(psn, f"PSN{s1}Function").apply


class VanillaPSN(nn.Module):
    """Borrowed from SpikingJelly.
    """

    def __init__(self, T: int, dtype):
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate.ATan()
        weight = torch.zeros([T, T]).to(dtype)
        bias = torch.zeros([T, 1]).to(dtype)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, 1.)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(-self.bias, self.weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 17)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='neuron_type',
        # possible values for `line_arg``
        line_vals=['torch', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (NCL=8*700)",
        args={"NCL": 8 * 700},
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['NCL'],
        # different possible values for `x_name`
        x_vals=[128 * i for i in range(1, 51)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='neuron_type',
        # possible values for `line_arg``
        line_vals=['torch', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (T=4)",
        args={"T": 4},
    ),
])
def bacnmark(T, NCL, neuron_type):
    x = torch.randn([T, NCL], device=DEVICE, dtype=DTYPE)
    grad_y = torch.randn_like(x)
    x.requires_grad = True

    results = 0, 0, 0
    if neuron_type == "torch":
        f = VanillaPSN(T, dtype=DTYPE).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "triton":
        f = get_psn_autograd_function(sg=SG)
        beta = torch.tensor(0.5, device=DEVICE, dtype=DTYPE, requires_grad=True)
        weight = torch.randn(
            [T, T],
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        bias = torch.randn(
            [T, 1],
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        results = triton.testing.do_bench(
            lambda: f(x, weight, bias).backward(grad_y), quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_psn", print_data=True, show_plots=True
    )
