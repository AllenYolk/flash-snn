import sys

sys.path.append("./")

import torch
import torch.nn as nn
import triton
from spikingjelly.activation_based import surrogate, neuron, functional

from flashsnn.ops import lif, spike_fn, flexsn, surrogate_kernels
from flashsnn import torch2triton

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


class SJLIF(neuron.LIFNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super().forward(x)
        functional.reset_net(self)
        return y


def get_lif_autograd_function():
    return getattr(lif, f"MultistepLIFHardDetachedFunction").apply


def lif_core(x: torch.Tensor, v: torch.Tensor):
    h = v*0.5 + x
    s = spike_fn(h - 1.)
    v = h * (1. - s.detach())
    return s, v


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
            'triton-flexsn',
        ],
        # label name for the lines
        line_names=[
            'Torch',
            'Torch (compile)',
            'SpikingJelly (CuPy)',
            'SpikingJelly (Torch)',
            'Triton',
            'Triton (flexsn)',
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
            'triton-flexsn',
        ],
        # label name for the lines
        line_names=[
            'Torch',
            'Torch (compile)',
            'SpikingJelly (CuPy)',
            'SpikingJelly (Torch)',
            'Triton',
            'Triton (flexsn)',
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
        f = VanillaLIF(beta=0.5, dtype=DTYPE).to(DEVICE)
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    if neuron_type == "torch-compile":
        f = torch.compile(
            VanillaLIF(beta=0.5, dtype=DTYPE).to(DEVICE), backend="inductor"
        )
        results = triton.testing.do_bench(
            lambda: f(x).backward(grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "triton":
        f = get_lif_autograd_function()
        results = triton.testing.do_bench(
            lambda: f(
                x, 0.5, surrogate_kernels.atan_surrogate_backward, False, False
            ).backward(grad_y),
            quantiles=QUANTILES
        )
    elif neuron_type == "triton-flexsn":
        core = lif_core
        graph = torch2triton.generate_inference_graph(
            core, (x, torch.randn_like(x))
        )
        fwd_graph, bwd_graph = torch2triton.generate_forward_and_backward_graph(
            core, (x, torch.randn_like(x)), requires_grad=(True, True)
        )
        info = flexsn.extract_info(
            fwd_graph, num_inputs=1, num_states=1, num_outputs=1
        )

        # prepare the inference kernel
        core_str, core_name = torch2triton.generate_triton_code_str(
            graph, core.__name__ + "_inference", verbose=False
        )
        f_inf = flexsn.get_flexsn_inference_kernel(
            core_str, core_name, info, verbose=False
        )
        # prepare the forward kernel
        core_str, core_name = torch2triton.generate_triton_code_str(
            fwd_graph, core.__name__ + "_forward", verbose=False
        )
        f_fwd = flexsn.get_flexsn_forward_kernel(
            core_str, core_name, info=info, verbose=False
        )
        # prepare the backward kernel
        core_str, core_name = torch2triton.generate_triton_code_str(
            bwd_graph, core.__name__ + "_backward", verbose=False
        )
        f_bwd = flexsn.get_flexsn_backward_kernel(
            core_str, core_name, info=info, verbose=False
        )

        f = flexsn.FlexSNFunction.apply
        results = triton.testing.do_bench(
            lambda: f(f_inf, f_fwd, f_bwd, info, x).backward(grad_y),
            quantiles=QUANTILES
        )
    elif neuron_type == "spikingjelly-cupy":
        f = SJLIF(
            tau=2.,
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
        f = SJLIF(
            tau=2.,
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
        save_path="./logs/benchmark_lif", print_data=True, show_plots=True
    )
