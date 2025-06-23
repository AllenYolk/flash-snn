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

print(torch2triton.template.get_spiking_neuron_inference_kernel("dota"))
print(
    torch2triton.template.get_spiking_neuron_forward_kernel(
        "monkey", ["foo", "bar", "nnn11"]
    )
)
print(
    torch2triton.template.get_spiking_neuron_backward_kernel(
        "dog", ["aa", "b", "ccc"]
    )
)
