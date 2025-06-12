import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate

from flashsnn.ops import lif
from flashsnn.utils import assert_close

BETA_LIST = [0.25 * i for i in range(0, 5)]
DETACH_RESET_LIST = [False, True]
SG_LIST = ["atan"]
SOFT_RESET_LIST = [False, True]
INPUT_SHAPE_LIST = [(4, 32, 3, 224, 224), (25, 4, 700)]
DTYPE_LIST = [torch.float32, torch.float16]


def get_lif_autograd_function(detach_reset: bool, sg: str, soft_reset: bool):
    if sg.lower() == "atan":
        s1 = "Atan"
    else:
        s1 = "Atan"

    if soft_reset:
        s2 = "Soft"
    else:
        s2 = "Hard"

    if detach_reset:
        s3 = "Detached"
    else:
        s3 = "NotDetached"

    return getattr(lif, f"MultistepLIF{s1}{s2}{s3}Function").apply


class VanillaLIF(nn.Module):

    def __init__(
        self, beta: float, detach_reset: bool, sg: str, soft_reset: bool,
        dtype: torch.dtype
    ):
        super().__init__()
        self.beta = torch.tensor(beta).to(dtype)
        self.one = torch.tensor(1.).to(dtype)
        self.detach_reset = detach_reset
        if sg.lower() == "atan":
            self.sg = surrogate.ATan()
        else:
            self.sg = surrogate.ATan()
        self.soft_reset = soft_reset

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


@pytest.mark.parametrize("beta", BETA_LIST)
@pytest.mark.parametrize("detach_reset", DETACH_RESET_LIST)
@pytest.mark.parametrize("sg", SG_LIST)
@pytest.mark.parametrize("soft_reset", SOFT_RESET_LIST)
@pytest.mark.parametrize("input_shape", INPUT_SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_lif_ops(beta, detach_reset, sg, soft_reset, input_shape, dtype):
    x_seq_1 = torch.randn(input_shape, device="cuda", dtype=dtype)
    x_seq_2 = x_seq_1.clone().detach()
    x_seq_1.requires_grad = True
    x_seq_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_seq_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = get_lif_autograd_function(detach_reset, sg, soft_reset)
    y1 = f1(x_seq_1, beta)
    y1.backward(grad_y_1)

    f2 = VanillaLIF(beta, detach_reset, sg, soft_reset, dtype).to("cuda")
    y2 = f2(x_seq_2)
    y2.backward(grad_y_2)

    assert_close(
        y1,
        y2,
        prefix="spike",
        ratio=0.03 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        x_seq_1.grad,
        x_seq_2.grad,
        prefix="x_seq.grad",
        ratio=0.03 if dtype == torch.float16 else 0.005,
    )
