import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate

from flashsnn.layers import FlexSN
from flashsnn.ops import spike_fn
from flashsnn.ops import lif, surrogate_kernels
from flashsnn.utils import assert_close

BETA_LIST = [0.5, 0.1, 0.9]
SHAPE_LIST = [(4, 5, 3, 224, 224), (11, 7, 700)]
DTYPE_LIST = [torch.float32, torch.float16]


def lif_core_generator(beta):

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        h = v*beta + x
        s = spike_fn(h - 1.)
        v = h * (1. - s.detach())
        return s, v

    return lif_core


@pytest.mark.parametrize("beta", BETA_LIST)
@pytest.mark.parametrize("shape", SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_flexsn_forward_backward(beta, shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    x1, x2 = x.clone(), x.clone()
    x1.requires_grad = True
    x2.requires_grad = True
    gs = torch.randn_like(x)

    core = lif_core_generator(beta=beta)

    # prepare graphs
    f = FlexSN(
        core, (x, torch.randn_like(x)),
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        requires_grad=(True, True)
    )
    s = f(x1)
    s.backward(gs)

    # handwritten LIF kernel
    ss = lif.MultistepLIFHardFunction.apply(
        x2, beta, surrogate_kernels.atan_surrogate_backward, True, False, False
    )
    ss.backward(gs)

    assert_close(s, ss, prefix="spike")
    assert_close(x1.grad, x2.grad, prefix="grad_x")


def strange_lif_core(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, rho: torch.Tensor
):
    h = 0.5*v + x
    s = spike_fn(h - (rho+1.))
    ss = spike_fn(h - 1.)
    rho = 0.99*rho + s
    v = h * (1.-s)
    vv = h * (1.-ss)
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
            v = h * (1.-s)
            vv = h * (1.-ss)
            sy = torch.sigmoid(y_seq[t])
            v = v*sy + vv * (1.-sy)
            s_seq[t] = s
            ss_seq[t] = ss
        return s_seq, ss_seq


@pytest.mark.parametrize("shape", SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_strange_flexsn_forward_backward(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    x1, x2 = x.clone(), x.clone()
    x1.requires_grad = True
    x2.requires_grad = True
    y = torch.randn(shape, dtype=dtype, device="cuda")
    y1, y2 = x.clone(), x.clone()
    y1.requires_grad = True
    y2.requires_grad = True
    gs = torch.randn_like(x)

    core = strange_lif_core

    # prepare graphs
    f = FlexSN(
        core,
        (x, y, torch.zeros_like(x), torch.zeros_like(x)),
        num_inputs=2,
        num_states=2,
        num_outputs=2,
        requires_grad=(True, True, True, True),
    )

    s1, s2 = f(x1, y1)
    s = s1 * s2
    s.backward(gs)

    ff = StrangeLIF()
    ss1, ss2 = ff(x2, y2)
    ss = ss1 * ss2
    ss.backward(gs)

    # there might bu numerical errors due to exponential operations.
    # adjust `ratio` in order to pass all tests!!
    assert_close(
        s1,
        ss1,
        prefix="spike1",
        ratio=0.015 if dtype == torch.float16 else 0.005
    )
    assert_close(
        s2,
        ss2,
        prefix="spike1",
        ratio=0.015 if dtype == torch.float16 else 0.005
    )
    assert_close(
        x1.grad,
        x2.grad,
        prefix="grad_x",
        ratio=0.015 if dtype == torch.float16 else 0.005
    )
    assert_close(
        y1.grad,
        y2.grad,
        prefix="grad_y",
        ratio=0.015 if dtype == torch.float16 else 0.005
    )


if __name__ == "__main__":
    test_flexsn_forward_backward(0.5, (4, 5, 3, 224, 224), torch.float16)
    test_strange_flexsn_forward_backward((4, 5, 3, 224, 224), torch.float16)
