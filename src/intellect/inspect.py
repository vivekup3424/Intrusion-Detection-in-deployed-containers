"""
Module with utility functions for reproducibility (seeds) and inspecting the status
of the objects and system.
"""
import io
import random
import sys

import numpy as np
import psutil


def set_seed(default=42) -> float:
    """Function for setting the seed for replicability.

    Args:
        default (int, optional): seed to be set. Defaults to 42.

    Returns:
        float: the seed set.
    """
    np.random.seed(default)
    random.seed(default)

    try:
        import torch
        torch.manual_seed(default)
    except ModuleNotFoundError:
        pass
    return default


def memory_stats() -> dict[str, object]:
    """Function to return memory stats, including GPU if available.

    Returns:
        dict[str, object]: dictionary with specifics
    """
    import torch
    return {'RAM': psutil.virtual_memory()._asdict(),
            'SWAP': psutil.swap_memory()._asdict(),
            'CUDA': torch.cuda.memory_stats() if torch.cuda.is_available() else {}}


def deep_get_size(obj: object, seen=None) -> int:
    """Recursively compute the size in memory of an object.

    Args:
        obj (object): the object of interest
        seen (_type_, optional): parameter to avoid duplicates. Defaults to None.

    Returns:
        int: the total size of the object.
    """

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(deep_get_size(v, seen) for v in obj.values())
        size += sum(deep_get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += deep_get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (io.TextIOBase, io.BufferedIOBase, io.RawIOBase,
                                                           io.IOBase, str, bytes, bytearray)):
        size += sum(deep_get_size(i, seen) for i in obj)

    return size
