import triton
import triton.language as tl


@triton.jit
def atan_surrogate_backward(h):
    sg = 3.141592653589793 * h
    sg = 1. / tl.fma(sg, sg, 1.)
    return sg.to(h.dtype)


@triton.jit
def sigmoid_surrogate_backward(h):
    # triton's exp() supports only fp32 and fp64, so we must convert it to fp32!
    sg = tl.sigmoid(h.to(tl.float32) * 4.)
    sg = 4. * sg * (1.-sg)
    return sg.to(h.dtype)
