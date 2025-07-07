import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate

from flashsnn.ops import bn
from flashsnn.utils import assert_close

DTYPE_LIST = [torch.float16, torch.float32]
T_LIST = [4, 7]
N_LIST = [11, 64]
C_LIST = [9, 128]
L_LIST = [13, 32]
MOMENTUM_LIST = [0.1, 0.5]
SOFT_RESET_LIST = [True, False]
DETACH_RESET_LIST = [True, False]


class VanillaLIF(nn.Module):

    def __init__(
        self, beta: float, dtype: torch.dtype, soft_reset: bool,
        detach_reset: bool
    ):
        super().__init__()
        self.beta = torch.tensor(beta).to(dtype)
        self.sg = surrogate.ATan()
        self.soft_reset = soft_reset
        self.detach_reset = detach_reset

    def forward(self, x_seq: torch.Tensor):
        v = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = self.beta * v + x_seq[t]
            s = self.sg(v - 1.)
            ss = s.detach() if self.detach_reset else s
            if self.soft_reset:
                v = v - ss
            else:
                v = v * (1.-ss)
            s_seq[t] = s
        return s_seq


class BNLIF(nn.Module):

    def __init__(self, C, momentum, soft_reset, detach_reset, dtype):
        super().__init__()
        # self.bn = nn.BatchNorm2d(C, momentum=momentum)
        self.bn = bn.BatchNorm2d(C, momentum=momentum)
        self.lif = VanillaLIF(
            beta=0.5,
            dtype=dtype,
            soft_reset=soft_reset,
            detach_reset=detach_reset
        )

    def forward(self, x_seq: torch.Tensor):
        out_shape = x_seq.shape
        x_seq = x_seq.flatten(0, 1)
        y_seq = self.bn(x_seq).reshape(out_shape)
        return self.lif(y_seq)


class BNResLIF(nn.Module):

    def __init__(self, C, momentum, soft_reset, detach_reset, dtype):
        super().__init__()
        # self.bn = nn.BatchNorm2d(C, momentum=momentum)
        self.bn = bn.BatchNorm2d(C, momentum=momentum)
        self.lif = VanillaLIF(
            beta=0.5,
            dtype=dtype,
            soft_reset=soft_reset,
            detach_reset=detach_reset
        )

    def forward(self, x_seq: torch.Tensor, r_seq: torch.Tensor):
        out_shape = x_seq.shape
        x_seq = x_seq.flatten(0, 1)
        y_seq = self.bn(x_seq).reshape(out_shape)
        return self.lif(y_seq + r_seq)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("C", C_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("soft_reset", SOFT_RESET_LIST)
@pytest.mark.parametrize("detach_reset", DETACH_RESET_LIST)
@pytest.mark.parametrize("momentum", MOMENTUM_LIST)
def test_bn_lif(T, N, C, L, momentum, soft_reset, detach_reset, dtype):
    x_1 = torch.randn([T, N, C, L, L], device="cuda", dtype=dtype)
    x_2 = x_1.clone().detach()
    x_1.requires_grad = True
    x_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = BNLIF(
        C,
        momentum=momentum,
        soft_reset=soft_reset,
        detach_reset=detach_reset,
        dtype=dtype,
    ).to("cuda")
    f2 = bn.BatchNorm2dLIF(
        C,
        momentum=momentum,
        soft_reset=soft_reset,
        detach_reset=detach_reset,
    ).to("cuda")
    f2.weight.data = f1.bn.weight.data
    f2.bias.data = f1.bn.bias.data

    y1 = f1(x_1)
    y1.backward(grad_y_1)

    y2 = f2(x_2)
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
        f1.bn.weight.grad,
        f2.weight.grad,
        prefix="weight.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        f1.bn.bias.grad,
        f2.bias.grad,
        prefix="bias.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )


@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("C", C_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("soft_reset", [False])
@pytest.mark.parametrize("detach_reset", [True])
@pytest.mark.parametrize("momentum", [0.1])
def test_bn_res_lif(T, N, C, L, momentum, soft_reset, detach_reset, dtype):
    x_1 = torch.randn([T, N, C, L, L], device="cuda", dtype=dtype)
    x_2 = x_1.clone().detach()
    x_1.requires_grad = True
    x_2.requires_grad = True
    r_1 = torch.randn([T, N, C, L, L], device="cuda", dtype=dtype)
    r_2 = r_1.clone().detach()
    r_1.requires_grad = True
    r_2.requires_grad = True
    grad_y_1 = torch.randn_like(x_1)
    grad_y_2 = grad_y_1.clone().detach()

    f1 = BNResLIF(
        C,
        momentum=momentum,
        soft_reset=soft_reset,
        detach_reset=detach_reset,
        dtype=dtype,
    ).to("cuda")
    f2 = bn.BatchNorm2dLIF(
        C,
        momentum=momentum,
        soft_reset=soft_reset,
        detach_reset=detach_reset,
    ).to("cuda")
    f2.weight.data = f1.bn.weight.data
    f2.bias.data = f1.bn.bias.data

    y1 = f1(x_1, r_1)
    y1.backward(grad_y_1)

    y2 = f2(x_2, r_2)
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
        r_1.grad,
        r_2.grad,
        prefix="res.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        f1.bn.weight.grad,
        f2.weight.grad,
        prefix="weight.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )
    assert_close(
        f1.bn.bias.grad,
        f2.bias.grad,
        prefix="bias.grad",
        ratio=0.04 if dtype == torch.float16 else 0.005,
    )


if __name__ == "__main__":
    test_bn_res_lif(
        T=3,
        N=1,
        C=4,
        L=2,
        momentum=0.1,
        dtype=torch.float16,
        soft_reset=True,
        detach_reset=False
    )
