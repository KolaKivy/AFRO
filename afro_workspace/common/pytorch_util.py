from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def _copy_to_cpu(obj):
    """
    Recursively copy a state_dict-like object to CPU.

    - Tensors are detached and moved to CPU.
    - dicts, lists and tuples are traversed recursively.
    - other objects are returned as-is.

    This is useful when saving checkpoints from a background thread
    to avoid holding references to CUDA memory.
    """
    if isinstance(obj, torch.Tensor):
        try:
            return obj.detach().cpu()
        except Exception:
            return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _copy_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_copy_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_copy_to_cpu(v) for v in obj)
    else:
        return obj
