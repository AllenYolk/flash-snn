import sys

sys.path.append("./")

import torch
import torch.nn as nn
import triton

from flashsnn.ops import leaky_integrator

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]
BETA = 0.5


class VanillaPLI(nn.Module):

    def __init__(self, beta_init: float, dtype: torch.dtype):
        super().__init__()
        self._beta = nn.Parameter(torch.tensor(beta_init).to(dtype))
        self.one = torch.tensor(1.).to(dtype)
        self.dtype = dtype

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    def forward(self, x_seq: torch.Tensor):
        y = torch.zeros_like(x_seq[0])
        y_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            y = self.beta * y + x_seq[t]
            y_seq[t] = y
        return y_seq


class ParallelPLI(nn.Module):

    def __init__(self, beta_init: float, dtype: torch.dtype):
        super().__init__()
        self._beta = nn.Parameter(torch.tensor(beta_init).to(dtype))
        self.one = torch.tensor(1.).to(dtype)
        self.dtype = dtype

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    def get_beta_gemm_weight(self, T: int) -> torch.Tensor:
        beta_gemm = torch.zeros([T, T], device=self._beta.device)
        # lower triangle: exponential
        for i in range(T):
            for j in range(i + 1):
                beta_gemm[i, j] = self.beta**(i - j)
        return beta_gemm

    def forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        beta_gemm = self.get_beta_gemm_weight(T)
        y_seq = torch.addmm(
            torch.tensor(0., device=x_seq.device, dtype=x_seq.dtype),
            beta_gemm,
            x_seq.flatten(1),
        ).view(x_seq.shape)
        return y_seq


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 17)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'torch_parallel', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Torch (parallel)', 'Triton'],
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
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=['torch', 'torch_parallel', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Torch (parallel)', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (T=4)",
        args={"T": 4},
    ),
])
def bacnmark(T, NCL, implementation):
    x = torch.randn([T, NCL], device=DEVICE, dtype=DTYPE)
    grad_y = torch.randn_like(x)
    x.requires_grad = True

    results = 0, 0, 0
    if implementation == "torch":
        f = VanillaPLI(beta_init=BETA, dtype=DTYPE).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif implementation == "torch_parallel":
        f = ParallelPLI(beta_init=BETA, dtype=DTYPE).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif implementation == "triton":
        f = leaky_integrator.MultistepPLIIterativeFunction.apply
        beta = torch.tensor(
            BETA, device=DEVICE, dtype=DTYPE, requires_grad=True
        )
        results = triton.testing.do_bench(
            lambda: f(
                x,
                torch.sigmoid(beta).expand(x.shape),
            ).backward(grad_y),
            quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_leaky_integrator",
        print_data=True,
        show_plots=True
    )
