import sys

sys.path.append("./")

import torch
import torch.fx as fx

from flashsnn import torch2triton
from flashsnn.ops import spike_fn


def lif_core_generator(beta):

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        h = v*beta + x
        s = spike_fn(h - 1.)
        v = h * (1.-s)
        return s, v

    return lif_core


def lif_core2(x: torch.Tensor, v: torch.Tensor, beta: torch.Tensor):
    h = v*beta + x
    s = spike_fn(h - 1.)
    v = h * (1.-s)
    return s, v


if __name__ == "__main__":

    lif_core = lif_core_generator(beta=0.5)
    traced = fx.symbolic_trace(lif_core)
    print("Forward Graph:")
    print(traced.graph)
    print("==" * 20)
    torch2triton.generate_backward_triton_code(
        lif_core, requires_grad=(True, True), verbose=True
    )

    traced = fx.symbolic_trace(lif_core2)
    print("Forward Graph:")
    print(traced.graph)
    print("==" * 20)
    torch2triton.generate_backward_triton_code(
        lif_core2, requires_grad=(True, True, False), verbose=True
    )
