import triton
import triton.language as tl


@triton.jit
def atan_surrogate_backward(h, dtype: tl.constexpr):
    pi = tl.full([1], 3.141592653589793, dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)

    sg = pi * h
    sg = (one / tl.fma(sg, sg, one)).to(dtype)
    return sg


@triton.jit
def sigmoid_surrogate_backward(h, dtype: tl.constexpr):
    four = tl.full([1], 4., dtype=dtype)
    one = tl.full([1], 1., dtype=dtype)

    # triton's exp() supports only fp32 and fp64, so we must convert it to fp32!
    sg = tl.sigmoid(h.to(tl.float32) * four)
    sg = four * sg * (one-sg)
    return sg.to(dtype)
