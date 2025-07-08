import pytest
import sys

sys.path.append("./")

import torch
import torch.nn as nn

from flashsnn.ops import ssa
from flashsnn.utils import assert_close

T_LIST = [4, 11]
N_LIST = [8, 13]
NUM_HEADS_LIST = [8, 13]
CPH_LIST = [16, 55]
L_LIST = [36, 81]
SCALE_LIST = [0.125, 0.99]
DTYPE_LIST = [torch.float32, torch.float16]


def ssa_torch(qkv, scale):
    # qkv.shape = [T, N, 3, NUM_HEADS, Cph, L]
    q = qkv[:, :, 0].transpose(-1, -2)
    k = qkv[:, :, 1].transpose(-1, -2)
    v = qkv[:, :, 2].transpose(-1, -2)

    o = k.transpose(-2, -1) @ v
    o = (q@o) * scale  # [T, N, NUM_HEADS, L, Cph]
    return o.transpose(-2, -1)


@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("NUM_HEADS", NUM_HEADS_LIST)
@pytest.mark.parametrize("Cph", CPH_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("scale", SCALE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_ssa_inference(T, N, NUM_HEADS, Cph, L, scale, dtype):
    qkv = torch.randn([T, N, 3, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")

    o1 = ssa.ssa_forward(qkv, scale)
    o2 = ssa_torch(qkv, scale)

    assert_close(o1, o2, prefix="ssa_output")


@pytest.mark.parametrize("T", T_LIST)
@pytest.mark.parametrize("N", N_LIST)
@pytest.mark.parametrize("NUM_HEADS", NUM_HEADS_LIST)
@pytest.mark.parametrize("Cph", CPH_LIST)
@pytest.mark.parametrize("L", L_LIST)
@pytest.mark.parametrize("scale", SCALE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_ssa_fwbw(T, N, NUM_HEADS, Cph, L, scale, dtype):
    qkv1 = torch.randn([T, N, 3, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    qkv2 = qkv1.clone()
    qkv1.requires_grad = True
    qkv2.requires_grad = True
    grad1 = torch.randn([T, N, NUM_HEADS, Cph, L], dtype=dtype).to("cuda")
    grad2 = grad1.clone()

    o1 = ssa.SSAFunction.apply(qkv1, scale)
    o1.backward(grad1)
    o2 = ssa_torch(qkv2, scale)
    o2.backward(grad2)

    gq1 = qkv1.grad[:, :, 0]
    gq2 = qkv2.grad[:, :, 0]
    gk1 = qkv1.grad[:, :, 1]
    gk2 = qkv2.grad[:, :, 1]
    gv1 = qkv1.grad[:, :, 2]
    gv2 = qkv2.grad[:, :, 2]

    assert_close(o1, o2, prefix="ssa_output")
    assert_close(gq1, gq2, prefix="q.grad")
    assert_close(gv1, gv2, prefix="v.grad")
    assert_close(gk1, gk2, prefix="k.grad")


if __name__ == "__main__":
    test_ssa_fwbw(
        T=11, N=8, NUM_HEADS=13, Cph=16, L=36, scale=0.125, dtype=torch.float32
    )
    test_ssa_fwbw(
        T=11, N=8, NUM_HEADS=13, Cph=16, L=36, scale=0.125, dtype=torch.float32
    )
