import torch.fx as fx


def extract_info(
    fwd_graph: fx.Graph,
    num_inputs: int = 1,
    num_states: int = 0,
    num_outputs: int = 1,
) -> dict:
    """Extract useful information from the forward graph.

    The forward core graph should have the following signature:
    [*inputs, *states] -> [*outputs, *states, *intermediates]

    The following information will be extracted:

    * fwd_core_returns: the return value names of the forward core. There might\
        be duplicated tensors in fwd_core_returns, but fwd_core_returns[2:] are\
        all unique!!!
    * num_fwd_core_returns: the length of fwd_core_returns
    * fwd_core_return_symbols: the names of the variables receiving the return 
        values of the forward core
    * fwd_kernel_returns: the return value names of the forward kernel; no
        duplicated tensors!!!
    * num_fwd_kernel_returns: the length of fwd_kernel_returns
    * extra_return_mapping: the mapping between the index i of fwd_core_returns[2:]\
        and the index j of fwd_kernel_returns. It can be used to map the return\
        values of the forward kernel to the input of the backward core.

    Args:
        fwd_graph (fx.Graph): the forward computational graph.
        num_inputs (int): Defaults to 1.
        num_states (int): Defaults to 1.
        num_outputs (int): Defaults to 1.

    Returns:
        dict
    """
    fwd_core_args = [n.name for n in fwd_graph.find_nodes(op="placeholder")]
    fwd_core_returns = []
    for n in fwd_graph.find_nodes(op="output"):
        for a in n.args[0]:
            fwd_core_returns.append(a.name)

    num_args = num_inputs + num_states
    num_inference_returns = num_outputs + num_states
    assert len(fwd_core_args) == num_args
    assert len(fwd_core_returns) >= num_inference_returns

    symbols = {}  # var in core -> var in kernel
    fwd_kernel_returns = []
    fwd_core_return_symbols = []
    for i, s in enumerate(fwd_core_returns[:num_outputs]):
        symbols[s] = f"s{i}"
        fwd_kernel_returns.append(f"s{i}")
        fwd_core_return_symbols.append(f"s{i}")
    for i, v in enumerate(fwd_core_returns[num_outputs:num_inference_returns]):
        symbols[v] = f"v{i}"
        fwd_core_return_symbols.append(f"v{i}")

    n = 0
    extra_return_mapping = []
    for ret in fwd_core_returns[num_inference_returns:]:  # intermediates
        if ret in symbols:  # duplicated core return detected
            fwd_core_return_symbols.append("_")
            sym = symbols[ret]
            if sym.startswith("v") and (sym not in fwd_kernel_returns):
                # return the state as an intermediate value!
                fwd_kernel_returns.append(sym)
        else:
            symbols[ret] = f"res{n}"
            fwd_kernel_returns.append(f"res{n}")
            fwd_core_return_symbols.append(f"res{n}")
            n += 1
        idx = fwd_kernel_returns.index(symbols[ret])  # locate the symbol
        extra_return_mapping.append(idx)

    assert (
        len(extra_return_mapping) == len(fwd_core_returns) -
        num_inference_returns
    )

    return {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "num_states": num_states,
        "fwd_core_inputs": fwd_core_args,
        "fwd_core_returns": fwd_core_returns,
        "num_fwd_core_returns": len(fwd_core_returns),
        "fwd_core_return_symbols": fwd_core_return_symbols,
        "fwd_kernel_returns": fwd_kernel_returns,
        "num_fwd_kernel_returns": len(fwd_kernel_returns),
        "extra_return_mapping": extra_return_mapping,
    }
