from functools import lru_cache, partial
from packaging import version

import torch


@lru_cache(maxsize=None)
def _check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


if _check_pytorch_version('2.4'):
    amp_custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    amp_custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    amp_custom_fwd = torch.cuda.amp.custom_fwd
    amp_custom_bwd = torch.cuda.amp.custom_bwd
