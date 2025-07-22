import sys

sys.path.append("./")

import torch
import torch.nn as nn
import torch._dynamo

torch._dynamo.config.suppress_errors = True

import triton
from spikingjelly.activation_based import surrogate

from flashsnn.ops import lif, surrogate_kernels
from flashsnn.layers import bn

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


class VanillaLIF(nn.Module):

    def __init__(self, beta: float, dtype: torch.dtype):
        super().__init__()
        self.beta = torch.tensor(beta).to(dtype)
        self.sg = surrogate.ATan()

    def forward(self, x_seq: torch.Tensor):
        v = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = self.beta * v + x_seq[t]
            s = self.sg(v - 1.)
            v = v * (1. - s.detach())
            s_seq[t] = s
        return s_seq


class BNLIF(nn.Module):

    def __init__(self, C):
        super().__init__()
        self.bn = nn.BatchNorm1d(C)
        self.lif = VanillaLIF(beta=0.5, dtype=DTYPE)

    def forward(self, x_seq: torch.Tensor):
        out_shape = x_seq.shape
        x_seq = x_seq.flatten(0, 1)
        y_seq = self.bn(x_seq).reshape(out_shape)
        return self.lif(y_seq)


class BNCompiledLIF(nn.Module):

    def __init__(self, C):
        super().__init__()
        self.bn = nn.BatchNorm1d(C)
        self.lif = torch.compile(VanillaLIF(beta=0.5, dtype=DTYPE))

    def forward(self, x_seq: torch.Tensor):
        out_shape = x_seq.shape
        x_seq = x_seq.flatten(0, 1)
        y_seq = self.bn(x_seq).reshape(out_shape)
        return self.lif(y_seq)


def get_triton_bn_lif_forward(C):

    def f(x_seq):
        f1 = bn.BatchNorm1d(C).to(DEVICE)
        f2 = lif.MultistepLIFHardFunction.apply
        out_shape = x_seq.shape
        x_seq = x_seq.flatten(0, 1)
        y_seq = f1(x_seq).reshape(out_shape)
        return f2(
            y_seq, 0.5, 1., surrogate_kernels.atan_surrogate_backward, True,
            False, False
        )

    return f


def get_triton_fused_bn_lif_forward(C):

    def f(x_seq):
        net = bn.BatchNorm1dLIF(C).to(DEVICE)
        return net(x_seq)

    return f


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[i for i in range(1, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=[
            'torch', 'torch-compile', 'torch-partly-compile', "triton",
            "triton-fused"
        ],
        # label name for the lines
        line_names=[
            'Torch', 'Torch (compile)', 'Torch (partly compile)', 'Triton',
            "Triton (fused)"
        ],
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
        plot_name="Performance (N=32, C=128, L=64*64)",
        args={
            "N": 32,
            "C": 128,
            "L": 64 * 64
        },
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['C'],
        # different possible values for `x_name`
        x_vals=[32 * i for i in range(1, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='implementation',
        # possible values for `line_arg``
        line_vals=[
            'torch', 'torch-compile', 'torch-partly-compile', "triton",
            "triton-fused"
        ],
        # label name for the lines
        line_names=[
            'Torch', 'Torch (compile)', 'Torch (partly compile)', 'Triton',
            "Triton (fused)"
        ],
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
        plot_name="Performance (T=4, N=64, L=64*64)",
        args={
            "T": 4,
            "N": 32,
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
        line_vals=[
            'torch', 'torch-compile', 'torch-partly-compile', "triton",
            "triton-fused"
        ],
        # label name for the lines
        line_names=[
            'Torch', 'Torch (compile)', 'Torch (partly compile)', 'Triton',
            "Triton (fused)"
        ],
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
        plot_name="Performance (T=4, N=32, C=128)",
        args={
            "T": 4,
            "N": 32,
            "C": 128
        },
    ),
])
def bacnmark(T, N, C, L, implementation):
    x = torch.randn([T, N, C, L], device=DEVICE, dtype=DTYPE)
    grad_y = torch.randn_like(x)
    x.requires_grad = True

    results = 0, 0, 0
    if implementation == "torch":
        f = BNLIF(C).to(DEVICE)
    elif implementation == "torch-compile":
        f = BNLIF(C).to(DEVICE)
        f = torch.compile(f, backend="inductor")
    elif implementation == "torch-partly-compile":
        f = BNCompiledLIF(C).to(DEVICE)
    elif implementation == "triton":
        f = get_triton_bn_lif_forward(C)
    elif implementation == "triton-fused":
        f = get_triton_fused_bn_lif_forward(C)
    results = triton.testing.do_bench(
        lambda: f(x).backward(grad_y), quantiles=QUANTILES
    )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_bnlif", print_data=True, show_plots=True
    )
