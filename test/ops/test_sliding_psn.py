import pytest
import sys

sys.path.append("./")

import torch

from flashsnn.ops import psn
from flashsnn.utils import assert_close

K_LIST = [3, 7, 13]
T_LIST = [11, 17, 23]
DTYPE_LIST = [torch.float32, torch.float16]


def gen_gemm_weight(input_weight: torch.Tensor, T: int, k: int):
    weight = torch.zeros([T, T], device=input_weight.device)
    for i in range(T):
        end = i + 1
        start = max(0, i + 1 - k)
        length = min(end - start, k)
        weight[i][start:end] = input_weight[k - length:k]
    return weight


@pytest.mark.parametrize("k", K_LIST)
@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_gen_sliding_psn_gemm_weight(k, T, dtype):
    input_weight1 = torch.randn([k], device="cuda", dtype=dtype)
    input_weight2 = input_weight1.clone().detach()
    input_weight1.requires_grad = True
    input_weight2.requires_grad = True
    grad1 = torch.randn([T, T], device="cuda", dtype=dtype)
    grad2 = grad1.clone().detach()

    weight1 = gen_gemm_weight(input_weight1, T, k)
    weight1.backward(grad1)

    weight2 = psn.GenerateSlidingPSNGemmWeightFunction.apply(input_weight2, T)
    weight2.backward(grad2)

    assert_close(weight1, weight2, prefix="gemm_weight")
    assert_close(
        input_weight1.grad, input_weight2.grad, prefix="gemm_weight.grad"
    )
