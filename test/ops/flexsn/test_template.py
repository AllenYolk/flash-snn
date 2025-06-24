import pytest
import sys

sys.path.append("./")

import torch

from flashsnn.ops import flexsn, torch2triton
from flashsnn.ops import lif
from flashsnn.utils import assert_close

BETA_LIST = [0.5, 0.1, 0.9]
SHAPE_LIST = [(4, 5, 3, 224, 224), (11, 7, 700)]
DTYPE_LIST = [torch.float32, torch.float16]


@torch.fx.wrap
def spike_fn(h):
    return (h >= 0.).to(h.dtype)


def lif_core_generator(beta):

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        h = v*beta + x
        s = spike_fn(h - 1.)
        v = h * (1.-s)
        # v = h * s
        return s, v

    return lif_core


@pytest.mark.parametrize("beta", BETA_LIST)
@pytest.mark.parametrize("shape", SHAPE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_flexsn_inference(beta, shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")

    core = lif_core_generator(beta=beta)
    core_str, core_name = torch2triton.generate_triton_code_str(
        core, verbose=True
    )
    f = flexsn.get_flexsn_inference_kernel(core_str, core_name, verbose=True)
    s = flexsn.flexsn_inference(x, f)

    ss = lif.MultistepLIFHardDetachedFunction.apply(x, beta)

    assert_close(s, ss, prefix="spike_lif")


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

    # prepare inference core
    core_str, core_name = torch2triton.generate_triton_code_str(
        core, verbose=True
    )
    f_inf = flexsn.get_flexsn_inference_kernel(
        core_str, core_name, verbose=True
    )

    # prepare forward core
    fwd_graph, bwd_graph = torch2triton.generate_backward_fx_graph(
        core, requires_grad=(True, True)
    )
    bi2fo = torch2triton.get_bi2fo(fwd_graph, bwd_graph)

    core_str, core_name = torch2triton.generate_triton_code_str(
        fwd_graph, core.__name__ + "_forward", verbose=True
    )
    f_fwd = flexsn.get_flexsn_forward_kernel(
        core_str, core_name, bi2fo=bi2fo, verbose=True
    )

    core_str, core_name = torch2triton.generate_triton_code_str(
        bwd_graph, core.__name__ + "_backward", verbose=True
    )
    f_bwd = flexsn.get_flexsn_backward_kernel(
        core_str, core_name, bi2fo=bi2fo, verbose=True
    )

    s = flexsn.FlexSNFunction.apply(x1, bi2fo, f_inf, f_fwd, f_bwd)
    s.backward(gs)

    ss = lif.MultistepLIFHardNotDetachedFunction.apply(x2, beta)
    ss.backward(gs)

    assert_close(s, ss, prefix="spike")
    assert_close(x1.grad, x2.grad, prefix="grad_x")


if __name__ == "__main__":
    # test_flexsn_inference(0.5, (4, 5, 3, 224, 224), torch.float32)
    test_flexsn_forward_backward(0.5, (4, 5, 3, 224, 224), torch.float32)
