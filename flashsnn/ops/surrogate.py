import triton
import triton.language as tl


@triton.jit
def atan_surrogate_backward(h, dtype: tl.constexpr):
    pi = tl.full([1], 3.141592653589793, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)

    sg = pi * h
    sg = (one / tl.fma(sg, sg, one)).to(dtype)
    return sg
