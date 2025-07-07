import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn

from flashsnn.ops import bn
from flashsnn.utils import assert_close

DTYPE_LIST = [torch.float16, torch.float32]
N_LIST = [11, 64, 256]
C_LIST = [9, 128, 256]
L_LIST = [7, 13, 32]
MOMENTUM_LIST = [0.1, 0.5, 0.9]


@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("C", C_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("momentum", MOMENTUM_LIST)
def test_bn(N, C, L, momentum, dtype):
    x_1 = torch.randn([N, C, L, L], device="cuda", dtype=dtype)
    x_2 = x_1.clone().detach()
    x_1.requires_grad = True
    x_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_1)
    grad_y_2 = grad_y_1.clone().detach()

    bn1 = bn.BatchNorm2d(C, momentum=momentum).to("cuda")
    bn2 = nn.BatchNorm2d(C, momentum=momentum).to("cuda")
    bn2.weight.data = bn1.weight.data
    bn2.bias.data = bn1.bias.data

    y1 = bn1(x_1)
    y1.backward(grad_y_1)

    y2 = bn2(x_2)
    y2.backward(grad_y_2)

    assert_close(
        y1,
        y2,
        prefix="output",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        x_1.grad,
        x_2.grad,
        prefix="input.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        bn1.weight.grad,
        bn2.weight.grad,
        prefix="weight.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        bn1.bias.grad,
        bn2.bias.grad,
        prefix="bias.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )


if __name__ == "__main__":
    test_bn(N=3, C=5, L=2, momentum=0.1, dtype=torch.float16)
