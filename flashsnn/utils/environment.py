"""Borrowed from:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
"""
from functools import lru_cache
import sys
from packaging import version

import triton
import torch

TRITON_VERSION_RECOMMENDED = "3.0.0"
PYTHON_VERSION_RECOMMENDED = "3.12"
TORCH_VERSION_RECOMMENDED = "2.0.1"


@lru_cache(maxsize=1)
def check_environments():
    """Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.
    """
    # Check Operating System
    if sys.platform == 'win32':
        print(
            "Detected Windows operating system. Triton does not have an "
            f"official Windows release. Please consider using a Linux "
            f"environment for compatibility."
        )

    triton_version = version.parse(triton.__version__)
    triton_version_recommended = version.parse(TRITON_VERSION_RECOMMENDED)
    if triton_version < triton_version_recommended:
        print(
            f"Current Triton version {triton_version} is below the "
            f"recommended {TRITON_VERSION_RECOMMENDED} version. "
            f"Please consider upgrading Triton."
        )

    python_version = version.parse(
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    python_version_recommended = version.parse(PYTHON_VERSION_RECOMMENDED)
    if python_version < python_version_recommended:
        print(
            f"Current Python version {python_version} is below the "
            f"recommended {PYTHON_VERSION_RECOMMENDED} version. "
            f"Please consider upgrading Python."
        )

    torch_version = version.parse(torch.__version__)
    torch_version_recommended = version.parse(TORCH_VERSION_RECOMMENDED)
    if torch_version < torch_version_recommended:
        print(
            f"Current PyTorch version {torch_version} is below the "
            f"recommended {TORCH_VERSION_RECOMMENDED} version. "
            f"Please consider upgrading PyTorch."
        )

    return None


check_environments()


@lru_cache(maxsize=1)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        print(
            "Triton is not supported on the current platform. Use CPU instead."
        )
        return "cpu"


@lru_cache(maxsize=1)
def get_platform() -> str:
    device = get_available_device()
    if device == 'cuda':
        return 'nvidia'
    elif device == 'hip':
        return 'amd'
    elif device == 'xpu':
        return 'intel'
    else:
        return device


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    info = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)
    return info['multiprocessor_count']


print("Streaming Multiprocessor Count: ", get_multiprocessor_count())
