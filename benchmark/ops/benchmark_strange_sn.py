import sys

sys.path.append("./")

import torch
import torch.nn as nn

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from spikingjelly.activation_based import surrogate
import triton

from flashsnn.ops import flexsn, spike_fn
from flashsnn import torch2triton

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


def strange_lif_core(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, rho: torch.Tensor
):
    h = 0.5*v + x
    s = spike_fn(h - (rho+1.))
    ss = spike_fn(h - 1.)
    rho = 0.99*rho + s
    v = h * (1. - s.detach())
    vv = torch.where(ss.to(bool) & s.to(torch.bool), h * (1.-ss), h - ss)
    sy = torch.sigmoid(y)
    v = v*sy + vv * (1.-sy)
    return s, ss, v, rho


class StrangeLIF(nn.Module):

    def __init__(self):
        super().__init__()
        self.sg = surrogate.ATan()

    def forward(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])
        rho = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        ss_seq = torch.empty_like(x_seq)
        for t in range(T):
            h = 0.5*v + x_seq[t]
            s = self.sg(h - (1.+rho))
            ss = self.sg(h - 1.)
            rho = 0.99*rho + s
            v = h * (1. - s.detach())
            vv = torch.where(
                ss.to(bool) & s.to(torch.bool), h * (1.-ss), h - ss
            )
            sy = torch.sigmoid(y_seq[t])
            v = v*sy + vv * (1.-sy)
            s_seq[t] = s
            ss_seq[t] = ss
        return s_seq, ss_seq


def run(f, x, y, g):

    def func():
        s1, s2 = f(x, y)
        s = s1 * s2
        s.backward(g)

    return func


def run_flexsn(f, x, y, g, f_inf, f_fwd, f_bwd, info):

    def func():
        s1, s2 = f(f_inf, f_fwd, f_bwd, info, x, y)
        s = s1 * s2
        s.backward(g)

    return func


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 7)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='neuron_type',
        # possible values for `line_arg``
        line_vals=[
            'torch',
            'torch-compile',
            'triton-flexsn',
        ],
        # label name for the lines
        line_names=[
            'Torch',
            'Torch (compile)',
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
            'triton-flexsn',
        ],
        # label name for the lines
        line_names=[
            'Torch',
            'Torch (compile)',
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
        args={"T": 8},
    ),
])
def bacnmark(T, NCL, neuron_type):
    x = torch.randn([T, NCL], device=DEVICE, dtype=DTYPE)
    y = torch.randn_like(x)
    grad_y = torch.randn_like(x)
    x.requires_grad = True
    y.requires_grad = True

    results = 0, 0, 0
    if neuron_type == "torch":
        f = StrangeLIF().to(DEVICE)
        results = triton.testing.do_bench(
            run(f, x, y, grad_y), quantiles=QUANTILES
        )
    if neuron_type == "torch-compile":
        f = torch.compile(StrangeLIF().to(DEVICE), backend="inductor")
        results = triton.testing.do_bench(
            run(f, x, y, grad_y), quantiles=QUANTILES
        )
    elif neuron_type == "triton-flexsn":
        core = strange_lif_core
        graph = torch2triton.generate_inference_graph(
            core, (x, y, torch.zeros_like(x), torch.zeros_like(x))
        )
        fwd_graph, bwd_graph = torch2triton.generate_forward_and_backward_graph(
            core, (x, y, torch.randn_like(x), torch.zeros_like(x)),
            requires_grad=(True, True, True, True)
        )
        info = flexsn.extract_info(
            fwd_graph, num_inputs=2, num_states=2, num_outputs=2
        )

        core_str, core_name = torch2triton.generate_triton_code_str(
            graph, core.__name__ + "_inference", verbose=False
        )
        f_inf = flexsn.get_flexsn_inference_kernel(
            core_str, core_name, info=info, verbose=False
        )

        core_str, core_name = torch2triton.generate_triton_code_str(
            fwd_graph, core.__name__ + "_forward", verbose=False
        )
        f_fwd = flexsn.get_flexsn_forward_kernel(
            core_str, core_name, info=info, verbose=False
        )

        # prepare backward core
        core_str, core_name = torch2triton.generate_triton_code_str(
            bwd_graph, core.__name__ + "_backward", verbose=False
        )
        f_bwd = flexsn.get_flexsn_backward_kernel(
            core_str, core_name, info=info, verbose=False
        )

        f = flexsn.FlexSNFunction.apply
        results = triton.testing.do_bench(
            run_flexsn(f, x, y, grad_y, f_inf, f_fwd, f_bwd, info),
            quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_strange_sn",
        print_data=True,
        show_plots=True
    )
