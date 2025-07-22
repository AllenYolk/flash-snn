import pytest
import sys

sys.path.append("./")

import torch
from spikingjelly.activation_based import surrogate

from flashsnn.ops import surrogate_kernels
from flashsnn.ops import qka
from flashsnn.utils import assert_close

T_LIST = [4, 11]
N_LIST = [8, 13]
NUM_HEADS_LIST = [8, 13]
CPH_LIST = [16, 55]
L_LIST = [36, 81]
DTYPE_LIST = [torch.float32, torch.float16]


def token_qka_torch(qk):
    # qk.shape = [T, N, 2, NUM_HEADS, Cph, L]
    q = qk[:, :, 0]
    k = qk[:, :, 1]

    q = torch.sum(q, dim=-2, keepdim=True)
    T = q.shape[0]
    attn = torch.empty_like(q)
    v = torch.zeros_like(q[0])
    sp = surrogate.ATan()  # surrogate function for LIF
    for t in range(T):
        h = 0.5*v + q[t]
        s = sp(h - 0.5)
        v = h * (1. - s.detach())
        attn[t] = s
    return k * attn


def channel_qka_torch(qk):
    # qk.shape = [T, N, 2, NUM_HEADS, Cph, L]
    q = qk[:, :, 0]
    k = qk[:, :, 1]

    q = torch.sum(q, dim=-1, keepdim=True)
    T = q.shape[0]
    attn = torch.empty_like(q)
    v = torch.zeros_like(q[0])
    sp = surrogate.ATan()  # surrogate function for LIF
    for t in range(T):
        h = 0.5*v + q[t]
        s = sp(h - 0.5)
        v = h * (1. - s.detach())
        attn[t] = s
    return k * attn


@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("NUM_HEADS", NUM_HEADS_LIST)
@pytest.mark.parametrize("Cph", CPH_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_token_qka_inference(T, N, NUM_HEADS, Cph, L, dtype):
    qk = torch.randn([T, N, 2, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    o1, h1 = qka.token_qka_forward(qk)
    o2 = token_qka_torch(qk)
    assert_close(o1, o2, prefix="token_qka_output")


@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("NUM_HEADS", NUM_HEADS_LIST)
@pytest.mark.parametrize("Cph", CPH_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_channel_qka_inference(T, N, NUM_HEADS, Cph, L, dtype):
    qk = torch.randn([T, N, 2, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    o1, h1 = qka.channel_qka_forward(qk)
    o2 = channel_qka_torch(qk)
    assert_close(o1, o2, prefix="channel_qka_output")


@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("NUM_HEADS", NUM_HEADS_LIST)
@pytest.mark.parametrize("Cph", CPH_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_token_qka_fwbw(T, N, NUM_HEADS, Cph, L, dtype):
    qk1 = torch.randn([T, N, 2, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    qk2 = qk1.clone()
    qk1.requires_grad = True
    qk2.requires_grad = True
    grad1 = torch.randn([T, N, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    grad2 = grad1.clone()

    o1 = qka.TokenQKAFunction.apply(
        qk1, surrogate_kernels.atan_surrogate_backward
    )
    o1.backward(grad1)
    o2 = token_qka_torch(qk2)
    o2.backward(grad2)

    gq1 = qk1.grad[:, :, 0]
    gq2 = qk2.grad[:, :, 0]
    gk1 = qk1.grad[:, :, 1]
    gk2 = qk2.grad[:, :, 1]

    assert_close(o1, o2, prefix="token_qka_output")
    assert_close(gq1, gq2, prefix="q.grad")
    assert_close(gk1, gk2, prefix="k.grad")


@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("NUM_HEADS", NUM_HEADS_LIST)
@pytest.mark.parametrize("Cph", CPH_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_channel_qka_fwbw(T, N, NUM_HEADS, Cph, L, dtype):
    qk1 = torch.randn([T, N, 2, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    qk2 = qk1.clone()
    qk1.requires_grad = True
    qk2.requires_grad = True
    grad1 = torch.randn([T, N, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    grad2 = grad1.clone()

    o1 = qka.ChannelQKAFunction.apply(
        qk1, surrogate_kernels.atan_surrogate_backward
    )
    o1.backward(grad1)
    o2 = channel_qka_torch(qk2)
    o2.backward(grad2)

    gq1 = qk1.grad[:, :, 0]
    gq2 = qk2.grad[:, :, 0]
    gk1 = qk1.grad[:, :, 1]
    gk2 = qk2.grad[:, :, 1]

    assert_close(o1, o2, prefix="channel_qka_output")
    assert_close(gk1, gk2, prefix="k.grad")
    assert_close(gq1, gq2, prefix="q.grad")


if __name__ == "__main__":
    # test_channel_qka_inference(
    # T=4, N=5, NUM_HEADS=7, Cph=13, L=5, dtype=torch.float32
    # )
    test_channel_qka_fwbw(
        T=1, N=1, NUM_HEADS=1, Cph=16, L=81, dtype=torch.float32
    )
