import torch


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


@torch.library.register_fake("flashsnn::spike_fn")
def _spike_fn_fake(h: torch.Tensor) -> torch.Tensor:
    return (h >= 0.).to(h.dtype)


def _setup_spike_fn_ctx(ctx, inputs, output):
    h, = inputs
    ctx.save_for_backward(h)


def _spike_fn_backward(ctx, grad_output: torch.Tensor):
    """The registered backward function for spike_fn, following the protocal 
    of torch.library.register_autograd().
    """
    h, = ctx.saved_tensors
    sg = torch.pi * h
    sg = torch.reciprocal(sg*sg + 1.).to(grad_output.dtype)
    return grad_output * sg


torch.library.register_autograd(
    "flashsnn::spike_fn", _spike_fn_backward, setup_context=_setup_spike_fn_ctx
)
