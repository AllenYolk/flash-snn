"""Borrowed from:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
"""
from typing import Callable
import functools
import contextlib

import torch


def continuous_and_device_guard(f: Callable) -> Callable:
    """Make sure all input tensors are contiguous and set to the same device.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous()
            for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        # find the first tensor in the argument list
        first_tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                first_tensor = arg
                break
        if first_tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    first_tensor = value
                    break
        if first_tensor is not None:
            ctx = torch.cuda.device(first_tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return f(*contiguous_args, **contiguous_kwargs)

    return wrapper
