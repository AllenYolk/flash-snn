"""Borrowed from:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
"""


def get_abs_err(x, y):
    """Max absolute error between two tensors.
    """
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    """Relative rooted squared error between two tensors.
    """
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base+1e-8)


def assert_close(x, y, ratio=0.005, prefix="", err_atol=1e-6):
    abs_atol = get_abs_err(x, y)
    err_ratio = get_err_ratio(x, y)
    msg = f"{prefix}: diff={abs_atol:.6f}, ratio={err_ratio:.6f}"

    if abs_atol <= err_atol:
        return
    assert err_ratio < ratio, msg
