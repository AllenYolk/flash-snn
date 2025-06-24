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


# as if it is a triton.JITFunction
def dota_iqweriop(x):
    return x + 1


def monkey_asdkli87(x):
    return x - 1


print(torch2triton.template.get_spiking_neuron_inference_kernel(dota_iqweriop))
print(
    torch2triton.template.get_spiking_neuron_forward_kernel(
        monkey_asdkli87, ["foo", "bar", "nnn11"]
    )
)
print(
    torch2triton.template.get_spiking_neuron_backward_kernel(
        monkey_asdkli87, ["aa", "b", "ccc"]
    )
)
