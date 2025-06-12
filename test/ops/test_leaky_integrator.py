import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn

from flashsnn.ops import leaky_integrator
from flashsnn.utils import assert_close

BETA_LIST = [0.25 * i for i in range(0, 5)]
INPUT_SHAPE_LIST = [(4, 32, 3, 224, 224), (25, 4, 700)]
DTYPE_LIST = [torch.float32, torch.float16]


class VanillaLI(nn.Module):

    def __init__(self, beta: float, dtype: torch.dtype):
        super().__init__()
        self.beta = torch.tensor(beta).to(dtype)
        self.one = torch.tensor(1.).to(dtype)

    def forward(self, x_seq: torch.Tensor):
        y = torch.zeros_like(x_seq[0])
        y_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            y = self.beta * y + x_seq[t]
            y_seq[t] = y
        return y_seq


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


@pytest.mark.parametrize("beta", BETA_LIST)
@pytest.mark.parametrize("input_shape", INPUT_SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_li_ops(beta, input_shape, dtype):
    x_seq_1 = torch.randn(input_shape, device="cuda", dtype=dtype)
    x_seq_2 = x_seq_1.clone().detach()
    x_seq_1.requires_grad = True
    x_seq_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_seq_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = leaky_integrator.MultistepLIIterativeFunction.apply
    y1 = f1(x_seq_1, beta)
    y1.backward(grad_y_1)

    f2 = VanillaLI(beta, dtype).to("cuda")
    y2 = f2(x_seq_2)
    y2.backward(grad_y_2)

    assert_close(
        y1,
        y2,
        prefix="y",
        ratio=0.03 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        x_seq_1.grad,
        x_seq_2.grad,
        prefix="x_seq.grad",
        ratio=0.03 if dtype == torch.float16 else 0.005,
    )


@pytest.mark.parametrize("beta_init", BETA_LIST)
@pytest.mark.parametrize("input_shape", INPUT_SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_pli_ops(beta_init, input_shape, dtype):
    x_seq_1 = torch.randn(input_shape, device="cuda", dtype=dtype)
    x_seq_2 = x_seq_1.clone().detach()
    x_seq_1.requires_grad = True
    x_seq_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_seq_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = leaky_integrator.MultistepPLIIterativeFunction.apply
    beta1 = torch.tensor(
        beta_init, device="cuda", dtype=dtype, requires_grad=True
    )
    y1 = f1(x_seq_1, torch.sigmoid(beta1).expand(x_seq_1.shape))
    y1.backward(grad_y_1)

    f2 = VanillaPLI(beta_init, dtype).to("cuda")
    y2 = f2(x_seq_2)
    y2.backward(grad_y_2)

    assert_close(
        y1,
        y2,
        prefix="y",
        ratio=0.05 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        x_seq_1.grad,
        x_seq_2.grad,
        prefix="x_seq.grad",
        ratio=0.05 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        beta1.grad,
        f2._beta.grad,
        prefix="beta.grad",
        ratio=0.1 if dtype == torch.float16 else 0.005,
    )
