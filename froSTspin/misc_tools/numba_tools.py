import numba
import numpy as np


def set_readonly_flag(*args):
    for a in args:
        a.flags["W"] = False


def set_writable_flag(*args):
    for a in args:
        a.flags["W"] = True


@numba.njit
def numba_find_indices(values, v0, nv):
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


@numba.njit
def numba_compute_tensor_strides(mat_strides, nrr, tensor_shape):
    NDIM = len(tensor_shape)
    tensor_strides = np.empty((NDIM,), dtype=np.int64)
    tensor_strides[NDIM - 1] = mat_strides[1]
    for i in range(1, NDIM - nrr):
        tensor_strides[NDIM - i - 1] = tensor_strides[NDIM - i] * tensor_shape[NDIM - i]
    tensor_strides[nrr - 1] = mat_strides[0]
    for i in range(1, nrr):
        tensor_strides[nrr - 1 - i] = tensor_strides[nrr - i] * tensor_shape[nrr - i]
    return tensor_strides


@numba.njit
def numba_transpose_reshape(old_mat, r1, r2, c1, c2, old_nrr, old_tensor_shape, perm):
    """
    numba version of
    old_mat[r1:r2, c1:c2].reshape(old_tensor_shape).transpose(perm)
    """
    NDIM = len(perm)

    old_tensor_strides = numba_compute_tensor_strides(
        old_mat.strides, old_nrr, old_tensor_shape
    )
    new_tensor_shape = np.empty((NDIM,), dtype=np.int64)
    new_tensor_strides = np.empty((NDIM,), dtype=np.int64)
    for i in range(NDIM):
        new_tensor_shape[i] = old_tensor_shape[perm[i]]
        new_tensor_strides[i] = old_tensor_strides[perm[i]]

    sht = numba.np.unsafe.ndarray.to_fixed_tuple(new_tensor_shape, NDIM)
    stridest = numba.np.unsafe.ndarray.to_fixed_tuple(new_tensor_strides, NDIM)
    permuted = np.lib.stride_tricks.as_strided(
        old_mat[r1:r2, c1:c2], shape=sht, strides=stridest
    )
    return permuted
