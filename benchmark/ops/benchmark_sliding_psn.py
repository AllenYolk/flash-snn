import sys

sys.path.append("./")

import torch
import triton

from flashsnn.ops import psn

DEVICE = "cuda"
DTYPE = torch.float32
QUANTILES = [0.5, 0.2, 0.8]


def gen_gemm_weight(input_weight: torch.Tensor, T: int, k: int):
    weight = torch.zeros([T, T], device=input_weight.device)
    for i in range(T):
        end = i + 1
        start = max(0, i + 1 - k)
        length = min(end - start, k)
        weight[i][start:end] = input_weight[k - length:k]
    return weight


@triton.testing.perf_report([
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 101)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='gen_gemm_weight_type',
        # possible values for `line_arg``
        line_vals=['torch', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (k=4)",
        args={"k": 10},
    ),
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4 * i for i in range(1, 101)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='gen_gemm_weight_type',
        # possible values for `line_arg``
        line_vals=['torch', 'triton'],
        # label name for the lines
        line_names=['Torch', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance (k=100)",
        args={"k": 100},
    ),
])
def bacnmark(T, k, gen_gemm_weight_type):
    input_weight = torch.randn([k], device="cuda", dtype=DTYPE)
    grad_y = torch.randn([T, T], device="cuda", dtype=DTYPE)
    input_weight.requires_grad = True

    results = 0, 0, 0
    if gen_gemm_weight_type == "torch":
        f = gen_gemm_weight
        results = triton.testing.do_bench(
            lambda: f(input_weight, T, k).backward(grad_y), quantiles=QUANTILES
        )
    elif gen_gemm_weight_type in ["triton"]:
        f = psn.GenerateSlidingPSNGemmWeightFunction.apply
        results = triton.testing.do_bench(
            lambda: f(input_weight, T).backward(grad_y), quantiles=QUANTILES
        )

    return results


if __name__ == "__main__":
    bacnmark.run(
        save_path="./logs/benchmark_sliding_psn",
        print_data=True,
        show_plots=True
    )
