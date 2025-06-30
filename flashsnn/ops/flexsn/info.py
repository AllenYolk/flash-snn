import torch.fx as fx


def extract_info(fwd_graph: fx.Graph) -> dict:
    """Extract useful information from the forward graph.

    fwd_core_returns: the return variable names of the forward core. There might
        be duplicated tensors in fwd_core_returns, but fwd_core_returns[2:] are
        all unique!!!
    N_fwd_core_returns: the length of fwd_core_returns
    fwd_kernel_returns: the return variable names of the forward kernel; no
        duplicated tensors!!!
    N_fwd_kernel_returns: the length of fwd_kernel_returns
    fwd_core_return_symbols: the names of the variables receiving the return 
        values of the forward core
    extra_return_mapping: the mapping between the indices i of fwd_core_returns[2:] 
        and the indice j of fwd_kernel_returns. It can be used to map the return
        values of the forward kernel to the input of the backward core.

    Args:
        fwd_graph (fx.Graph)

    Returns:
        dict
    """
    fwd_core_returns = []
    for n in fwd_graph.find_nodes(op="output"):
        for a in n.args[0]:
            fwd_core_returns.append(a.name)

    symbols = {fwd_core_returns[0]: "s", fwd_core_returns[1]: "v"}
    cnt = {"s": 1, "v": 1}
    n = 0
    fwd_kernel_returns = ["s"]
    fwd_core_return_symbols = ["s", "v"]

    for ret in fwd_core_returns[2:]:
        if ret in symbols:
            fwd_core_return_symbols.append(
                symbols[ret] + f"_{cnt[symbols[ret]]}"
            )  # dummy var; this var will never be used!
            cnt[symbols[ret]] += 1
            if symbols[ret] == "v" and cnt["v"] == 2:
                # we need to return v
                fwd_kernel_returns.append("v")
        else:
            symbols[ret] = f"res{n}"
            cnt[f"res{n}"] = 1
            fwd_kernel_returns.append(f"res{n}")
            fwd_core_return_symbols.append(f"res{n}")
            n += 1

    extra_return_mapping = []
    for ct in fwd_core_returns[2:]:
        idx = fwd_kernel_returns.index(symbols[ct])
        extra_return_mapping.append(idx)

    return {
        "fwd_core_returns": fwd_core_returns,
        "N_fwd_core_returns": len(fwd_core_returns),
        "fwd_kernel_returns": fwd_kernel_returns,
        "N_fwd_kernel_returns": len(fwd_kernel_returns),
        "fwd_core_return_symbols": fwd_core_return_symbols,
        "extra_return_mapping": extra_return_mapping,
    }
