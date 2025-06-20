import pytest
import sys

sys.path.append("./")

import torch
import torch.fx as fx
import triton
import triton.language as tl

from flashsnn.ops import torch2triton
from flashsnn.ops import surrogate_kernels, lif
from flashsnn.utils import assert_close, type_dict

if __name__ == "__main__":

    def lif_core(x: torch.Tensor, v: torch.Tensor, beta: torch.Tensor):
        h = v*beta + x
        s = torch2triton.spike_fn(h, 1.)
        v = h * (1.-s)
        return s, v

    traced = fx.symbolic_trace(lif_core)
    print("Forward Graph:")
    print(traced.graph)
    print("==" * 20)

    backward_triton_code = torch2triton.generate_backward_triton_code(
        lif_core, requires_grad=(True, True, False, False), verbose=True
    )
