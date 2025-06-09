import math
import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate

from flashsnn.ops import psn
from flashsnn.utils import assert_close

SG_LIST = ["atan"]
BACKEND_LIST = ["triton", "torch"]
INPUT_SHAPE_LIST = [(4, 32, 3, 224, 224), (5, 4, 700)]
DTYPE_LIST = [torch.float32, torch.float16]

torch.manual_seed(2025)


def get_psn_autograd_function(sg: str, backend):
    if sg.lower() == "atan":
        s1 = "Atan"
    else:
        s1 = "Atan"
    if backend == "triton":
        s2 = ""
    else:
        s2 = "Torch"
    return getattr(psn, f"PSN{s1}{s2}Function").apply


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


@pytest.mark.parametrize("sg", SG_LIST)
@pytest.mark.parametrize("input_shape", INPUT_SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("backend", BACKEND_LIST)
def test_lif_ops(sg, input_shape, dtype, backend):
    x_seq_1 = torch.randn(input_shape, device="cuda", dtype=dtype)
    x_seq_2 = x_seq_1.clone().detach()
    x_seq_1.requires_grad = True
    x_seq_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_seq_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = VanillaPSN(T=input_shape[0], dtype=dtype).to("cuda")
    weight2 = f1.weight.data.clone().detach()
    bias2 = f1.bias.data.clone().detach()
    weight2.requires_grad = True
    bias2.requires_grad = True
    y1 = f1(x_seq_1)
    y1.backward(grad_y_1)

    f2 = get_psn_autograd_function(sg, backend)
    y2 = f2(x_seq_2, weight2, bias2)
    y2.backward(grad_y_2)

    assert_close(
        y1,
        y2,
        prefix="spike",
        ratio=0.05 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        x_seq_1.grad,
        x_seq_2.grad,
        prefix="x_seq.grad",
        ratio=0.05 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        f1.bias.grad,
        bias2.grad,
        prefix="bias.grad",
        ratio=0.05 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        f1.weight.grad,
        weight2.grad,
        prefix="weight.grad",
        ratio=0.05 if dtype == torch.float16 else 0.005,
    )
