from typing import Callable, Tuple, Optional
import math

import torch
import torch.nn as nn

from flashsnn.ops import flexsn as flexsn_ops
from flashsnn import torch2triton

__all__ = ["FlexSN"]


class FlexSN(nn.Module):

    def __init__(
        self,
        core: Callable,
        example_inputs: Tuple[torch.Tensor],
        num_inputs: int,
        num_states: int,
        num_outputs: int,
        requires_grad: Optional[Tuple[bool]] = None,
    ):
        """FlexSN: customized spiking neuron layer with Triton acceleration.

        Args:
            core (Callable): a function describing the single-step inference
                dynamics of the spiking neuron. It should have the following 
                signature: [*inputs, *states] -> [*outputs, *states], and the 
                arguments and return values should all be tensors.
            example_inputs (Tuple[torch.Tensor]): example inputs to `core` 
                with the form of [*inputs, *states]. 
            num_inputs (int): number of inputs. It should strictly match the 
                number of "inputs" in `core`'s arguments and `example_inputs`.
            num_states (int): number of states. It should strictly match the 
                number of "states" in `core`'s arguments, `core`'s return values,
                and `example_inputs`.
            num_outputs (int): number of outputs. It should strictly match the 
                number of "outputs" in `core`'s return values.
            requires_grad (Optional[Tuple[bool]], optional): whether the input
                tensors (i.e. [*inputs, *states]) requires gradients. This info
                is used to generate the forward and backward graphs. Its length
                should match the number of `core`'s arguments and the length of
                `example_inputs`. If None, it will be set at the `requires_grad` 
                attributes of `example_inputs`. Defaults to None.
        """
        super().__init__()
        self.core = core
        self.inf_graph = torch2triton.generate_inference_graph(
            core, example_inputs
        )
        self.fwd_graph, self.bwd_graph = torch2triton.generate_forward_and_backward_graph(
            core, example_inputs, requires_grad=requires_grad
        )
        self.info = flexsn_ops.extract_info(
            self.fwd_graph, num_inputs, num_states, num_outputs
        )
        self.num_inputs = num_inputs
        self.num_states = num_states
        self.num_outputs = num_outputs

        core_str, core_name = torch2triton.generate_triton_code_str(
            self.inf_graph, core.__name__ + "_inference"
        )
        self.f_inf = flexsn_ops.get_flexsn_inference_kernel(
            core_str, core_name, info=self.info
        )

        core_str, core_name = torch2triton.generate_triton_code_str(
            self.fwd_graph, core.__name__ + "_forward"
        )
        self.f_fwd = flexsn_ops.get_flexsn_forward_kernel(
            core_str, core_name, info=self.info
        )

        core_str, core_name = torch2triton.generate_triton_code_str(
            self.bwd_graph, core.__name__ + "_backward"
        )
        self.f_bwd = flexsn_ops.get_flexsn_backward_kernel(
            core_str, core_name, info=self.info
        )

    def forward(self, *inputs):
        return flexsn_ops.FlexSNFunction.apply(
            self.f_inf,
            self.f_fwd,
            self.f_bwd,
            self.info,
            *inputs,
        )

    def extra_repr(self):
        return (
            f"core={self.core.__name__}, "
            f"num_inputs={self.num_inputs}, "
            f"num_states={self.num_states}, "
            f"num_outputs={self.num_outputs}, "
        )
