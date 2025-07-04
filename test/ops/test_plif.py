import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate

from flashsnn.ops import plif, surrogate_kernels
from flashsnn.utils import assert_close

BETA_INIT_LIST = [-0.4, -0.1, 0., 0.1, 0.4]
DETACH_RESET_LIST = [False, True]
SOFT_RESET_LIST = [False, True]
INPUT_SHAPE_LIST = [(4, 32, 3, 224, 224), (25, 4, 700)]
DTYPE_LIST = [torch.float32, torch.float16]
INPLACE_LIST = [True, False]

torch.manual_seed(2025)


def get_plif_autograd_function(soft_reset: bool):
    if soft_reset:
        s2 = "Soft"
    else:
        s2 = "Hard"

    return getattr(plif, f"MultistepPLIF{s2}Function").apply


class VanillaPLIF(nn.Module):

    def __init__(
        self, beta_init: float, detach_reset: bool, soft_reset: bool,
        dtype: torch.dtype
    ):
        super().__init__()
        self._beta = nn.Parameter(torch.tensor(beta_init).to(dtype))
        self.one = torch.tensor(1.).to(dtype)
        self.detach_reset = detach_reset
        self.sg = surrogate.ATan()
        self.soft_reset = soft_reset
        self.dtype = dtype

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    def forward(self, x_seq: torch.Tensor):
        v = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = self.beta * v + x_seq[t]
            s = self.sg(v - self.one)
            if self.soft_reset:
                if self.detach_reset:
                    v = v - s.detach()
                else:
                    v = v - s
            else:
                if self.detach_reset:
                    v = v * (self.one - s.detach())
                else:
                    v = v * (self.one - s)
            s_seq[t] = s
        return s_seq


@pytest.mark.parametrize("beta_init", BETA_INIT_LIST)
@pytest.mark.parametrize("detach_reset", DETACH_RESET_LIST)
@pytest.mark.parametrize("soft_reset", SOFT_RESET_LIST)
@pytest.mark.parametrize("input_shape", INPUT_SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("inplace", INPLACE_LIST)
def test_lif_ops(
    beta_init, detach_reset, soft_reset, input_shape, dtype, inplace
):
    x_seq_1 = torch.randn(input_shape, device="cuda", dtype=dtype)
    x_seq_2 = x_seq_1.clone().detach()
    x_seq_1.requires_grad = True
    x_seq_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_seq_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = get_plif_autograd_function(soft_reset)
    beta1 = torch.tensor(
        beta_init, device="cuda", dtype=dtype, requires_grad=True
    )
    y1 = f1(
        x_seq_1, beta1.expand(x_seq_1.shape),
        surrogate_kernels.atan_surrogate_backward, detach_reset, inplace,
        inplace
    )
    y1.backward(grad_y_1)

    f2 = VanillaPLIF(beta_init, detach_reset, soft_reset, dtype).to("cuda")
    y2 = f2(x_seq_2)
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
        beta1.grad,
        f2._beta.grad,
        prefix="beta.grad",
        ratio=0.1 if dtype == torch.float16 else 0.005,
    )
