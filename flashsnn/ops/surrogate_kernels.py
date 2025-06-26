import torch
import triton
import triton.language as tl


@torch.library.custom_op("flashsnn::spike_fn", mutates_args=())
def spike_fn(h: torch.Tensor) -> torch.Tensor:
    """Spike generation.

    The function is registered as a custom pytorch operator so that fx will not 
    trace through this function!

    According to pytorch docs, reasons for creating a custom op include: 
    * Wrapping a third-party library or custom kernel to work with PyTorch 
        subsystems like Autograd. 
    * Preventing torch.compile/export/FX tracing from peeking 
        inside your function.
    """
    return (h >= 0.).to(h.dtype)


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
