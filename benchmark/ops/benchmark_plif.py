import sys

sys.path.append("./")

import torch
import torch.nn as nn
import triton
from spikingjelly.activation_based import surrogate, functional, neuron

from flashsnn.ops import plif, surrogate_kernels

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


def get_plif_autograd_function():
    return getattr(plif, f"MultistepPLIFHardFunction").apply


class VanillaPLIF(nn.Module):

    def __init__(self, beta_init: float, dtype: torch.dtype):
        super().__init__()
        self._beta = nn.Parameter(torch.tensor(beta_init).to(dtype))
        self.one = torch.tensor(1.).to(dtype)
        self.sg = surrogate.ATan()
        self.dtype = dtype

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    def forward(self, x_seq: torch.Tensor):
        v = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = self.beta * v + x_seq[t]
            s = self.sg(v - self.one)
            v = v * (self.one - s.detach())
            s_seq[t] = s
        return s_seq


class SJPLIF(neuron.ParametricLIFNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super().forward(x)
        functional.reset_net(self)
        return y


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='neuron_type',
        # possible values for `line_arg``
        line_vals=[
            'torch',
            'torch-compile',
            'spikingjelly-cupy',
            'spikingjelly-torch',
            'triton',
        ],
        # label name for the lines
        line_names=[
            'Torch',
            'Torch (compile)',
            'SpikingJelly (CuPy)',
            'SpikingJelly (Torch)',
            'Triton',
        ],
        # line styles
        styles=[
            ('green', ':'),
            ('blue', '--'),
            ('cyan', '-.'),
            ('orange', ':'),
            ('red', '-'),
            ('red', "--"),
        ],
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
        line_vals=[
            'torch',
            'torch-compile',
            'spikingjelly-cupy',
            'spikingjelly-torch',
            'triton',
        ],
        # label name for the lines
        line_names=[
            'Torch',
            'Torch (compile)',
            'SpikingJelly (CuPy)',
            'SpikingJelly (Torch)',
            'Triton',
        ],
        # line styles
        styles=[
            ('green', ':'),
            ('blue', '--'),
            ('cyan', '-.'),
            ('orange', ':'),
            ('red', '-'),
            ('red', "--"),
        ],
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
        f = VanillaPLIF(beta_init=0.5, dtype=DTYPE).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "torch-compile":
        f = torch.compile(
            VanillaPLIF(beta_init=0.5, dtype=DTYPE).to(DEVICE),
            backend="inductor"
        )
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "triton":
        f = get_plif_autograd_function()
        beta = torch.tensor(0.5, device=DEVICE, dtype=DTYPE, requires_grad=True)
        results = triton.testing.do_bench(
            lambda: f(
                x, beta.expand(x.shape), surrogate_kernels.
                atan_surrogate_backward, True, False, False
            ).backward(grad_y),
            quantiles=QUANTILES
        )
    elif neuron_type == "spikingjelly-cupy":
        f = SJPLIF(
            init_tau=2.,
            decay_input=False,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
            backend="cupy"
        ).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "spikingjelly-torch":
        f = SJPLIF(
            init_tau=2.,
            decay_input=False,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
            backend="torch"
        ).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_plif", print_data=True, show_plots=True
    )
