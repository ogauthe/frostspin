import numba
import numpy as np


@numba.njit
def _numba_find_indices(values, v0, nv):
    """
    Find the nv indices where values == v0.

    This is equivalent to (values == v0).nonzero()[0].view(np.uint64), but we benefit
    from knowing nv beforehand. If nv is smaller than the actual number of values, only
    the first nv ones will be returned. If nv is larger than this number, the function
    will crash.

    Parameters
    ----------
    values : 1D int ndarray
        Array whose matching indices must be found.
    v0 : int
        Value to look for.
    nv : number of values

    Returns
    -------
    indices : (nv,) uint64 ndarray
        Array of indices matching v0.
    """
    indices = np.empty((nv,), dtype=np.uint64)
    j = 0
    i = 0
    while j < nv:
        if values[i] == v0:
            indices[j] = i
            j += 1
        i += 1
    return indices


def set_readonly_flag(*args):
    for a in args:
        a.flags["W"] = False


def set_writable_flag(*args):
    for a in args:
        a.flags["W"] = True
