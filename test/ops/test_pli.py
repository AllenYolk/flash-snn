import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn

from flashsnn.ops import pli
from flashsnn.utils import assert_close

BETA_LIST = [0.25 * i for i in range(0, 5)]
INPUT_SHAPE_LIST = [(4, 32, 3, 224, 224), (25, 4, 700)]
DTYPE_LIST = [torch.float32, torch.float16]
INPLACE_LIST = [True, False]


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


@pytest.mark.parametrize("beta_init", BETA_LIST)
@pytest.mark.parametrize("input_shape", INPUT_SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("inplace", INPLACE_LIST)
def test_pli_ops(beta_init, input_shape, dtype, inplace):
    x_seq_1 = torch.randn(input_shape, device="cuda", dtype=dtype)
    x_seq_2 = x_seq_1.clone().detach()
    x_seq_1.requires_grad = True
    x_seq_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_seq_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = pli.MultistepPLIFunction.apply
    beta1 = torch.tensor(
        beta_init, device="cuda", dtype=dtype, requires_grad=True
    )
    y1 = f1(x_seq_1, beta1.expand(x_seq_1.shape), inplace, inplace)
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
