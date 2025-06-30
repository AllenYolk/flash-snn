import triton
import triton.language as tl


@triton.jit
def atan_surrogate_backward(h, dtype: tl.constexpr):
    one = tl.full([1], 1., dtype=dtype)
    sg = 3.141592653589793 * h
    sg = (one / tl.fma(sg, sg, one)).to(dtype)
    return sg


@triton.jit
def sigmoid_surrogate_backward(h, dtype: tl.constexpr):
    # triton's exp() supports only fp32 and fp64, so we must convert it to fp32!
    sg = tl.sigmoid(h.to(tl.float32) * 4.)
    sg = 4. * sg * (1.-sg)
    return sg.to(dtype)
