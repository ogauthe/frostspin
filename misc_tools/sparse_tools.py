import numba
import numpy as np
import scipy.sparse as ssp


def custom_sparse_product(sparse_left, dense_mid, sparse_right):
    """
    Compute sparse_left @ dense_mid @ sparse_right
    """
    assert dense_mid.shape == (sparse_left.shape[1], sparse_right.shape[0])
    sl = sparse_left.tocsr()
    sr = sparse_right.tocsc()
    return _numba_custom_sparse(
        sl.shape[0],
        sr.shape[1],
        dense_mid,
        sl.indptr,
        sl.indices,
        sl.data,
        sr.indptr,
        sr.indices,
        sr.data,
    )


@numba.njit(parallel=True)
def _numba_custom_sparse(
    nr, nc, dense, sl_pointer, sl_indices, sl_nnz, sr_pointer, sr_indices, sr_nnz
):
    res = np.empty((nr, nc), dtype=dense.dtype)
    for i in numba.prange(nr):
        for j in numba.prange(nc):
            c = 0.0
            for m in range(sl_pointer[i], sl_pointer[i + 1]):
                for n in range(sr_pointer[j], sr_pointer[j + 1]):
                    c += sl_nnz[m] * dense[sl_indices[m], sr_indices[n]] * sr_nnz[n]
            res[i, j] = c
    return res


def custom_double_sparse_product(sl1, d1, sr1, sl2, d2, sr2):
    """
    Compute sl1 @ d1 @ sr1 + sl2 @ d2 @ sr2
    """
    assert d1.shape == (sl1.shape[1], sr1.shape[0]), "shape mismatch in matrix product"
    assert d2.shape == (sl2.shape[1], sr2.shape[0]), "shape mismatch in matrix product"
    assert sl1.shape[0] == sl2.shape[0]
    assert sr1.shape[1] == sr2.shape[1]
    assert d1.dtype == d2.dtype
    sl1 = sl1.tocsr()
    sr1 = sr1.tocsc()
    sl2 = sl2.tocsr()
    sr2 = sr2.tocsc()
    return _numba_double_custom_sparse(
        sl1.shape[0],
        sr1.shape[1],
        d1,
        d2,
        sl1.indptr,
        sl1.indices,
        sl1.data,
        sr1.indptr,
        sr1.indices,
        sr1.data,
        sl2.indptr,
        sl2.indices,
        sl2.data,
        sr2.indptr,
        sr2.indices,
        sr2.data,
    )


@numba.njit(parallel=True)
def _numba_double_custom_sparse(
    nrow,
    ncol,
    d1,
    d2,
    sl1_pointer,
    sl1_indices,
    sl1_nnz,
    sr1_pointer,
    sr1_indices,
    sr1_nnz,
    sl2_pointer,
    sl2_indices,
    sl2_nnz,
    sr2_pointer,
    sr2_indices,
    sr2_nnz,
):
    res = np.empty((nrow, ncol), dtype=d1.dtype)
    for i in numba.prange(nrow):
        for j in numba.prange(ncol):
            c = 0.0
            for m in range(sl1_pointer[i], sl1_pointer[i + 1]):
                for n in range(sr1_pointer[j], sr1_pointer[j + 1]):
                    c += sl1_nnz[m] * d1[sl1_indices[m], sr1_indices[n]] * sr1_nnz[n]
            for m in range(sl2_pointer[i], sl2_pointer[i + 1]):
                for n in range(sr2_pointer[j], sr2_pointer[j + 1]):
                    c += sl2_nnz[m] * d2[sl2_indices[m], sr2_indices[n]] * sr2_nnz[n]
            res[i, j] = c
    return res


def sparse_transpose(m, shape, axes, n_row_axes, copy=False):
    """
    Consider a sparse matrix as a higher rank tensor, transpose its axes and return a
    new sparse matrix.

    Parameters
    ----------
    m : sparse array
        Sparse matrix to transpose.
    shape : tuple of ints
        Input shape when viewed as a tensor.
    axes : tuple of ints
        New position for the axes after transpose.
    n_row_axes : int
        Number of axes to be merged as row in new matrix.
    copy : bool
        Whether to copy data. Default is False.

    Returns
    -------
    out : coo_array
        Transposed tensor cast as a coo_array.
    """
    if sorted(axes) != list(range(len(shape))):
        raise ValueError("axes do not match shape")
    size = m.shape[0] * m.shape[1]
    if np.prod(shape) != size:
        raise ValueError("invalid tensor shape")

    strides1 = np.array([1, *shape[:0:-1]]).cumprod()[::-1]
    strides2 = np.array([1, *[shape[i] for i in axes[:0:-1]]]).cumprod()
    nsh = (size // strides2[n_row_axes], strides2[n_row_axes])
    # swap strides instead of swapping the full array of tensor indices
    strides2 = strides2[-np.argsort(axes) - 1].copy()
    mcoo = m.tocoo()
    ind1D = np.ravel_multi_index((mcoo.row, mcoo.col), mcoo.shape)
    ind1D = (ind1D[:, None] // strides1 % shape) @ strides2
    row, col = np.unravel_index(ind1D, nsh)
    out = ssp.coo_array((m.data, (row, col)), shape=nsh, copy=copy)
    return out


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


@numba.njit
def _numba_get_strides(shape, axes, n_leg_rows):
    ndim = len(axes)
    rstrides1 = np.ones((n_leg_rows,), dtype=np.int64)
    rstrides1[1:] = shape[n_leg_rows - 1 : 0 : -1]
    rstrides1 = rstrides1.cumprod()[::-1].copy()
    rmod = shape[:n_leg_rows]

    cstrides1 = np.ones((ndim - n_leg_rows,), dtype=np.int64)
    cstrides1[1:] = shape[-1:n_leg_rows:-1]
    cstrides1 = cstrides1.cumprod()[::-1].copy()
    cmod = shape[n_leg_rows:]

    new_strides = np.ones((ndim,), dtype=np.int64)
    for i in range(ndim - 1, 0, -1):
        new_strides[axes[i - 1]] = new_strides[axes[i]] * shape[axes[i]]
    rstrides2 = new_strides[:n_leg_rows]
    cstrides2 = new_strides[n_leg_rows:]
    ncol = cmod.prod()
    nci = (np.arange(ncol).reshape(-1, 1) // cstrides1 % cmod * cstrides2).sum(axis=1)
    return rstrides1, rstrides2, rmod, nci


@numba.njit(parallel=True)
def _numba_parallel_strided_transpose(m, shape, axes, n_leg_rows, rshift=0, cshift=0):
    """
    Decompose a given matrix as a tensor, swap it axes according to perm and return a
    contiguous 1D array. It is possible to consider only a (non-contiguous) submatrix
    with rshift and cshift. Even with cshift=0, the submatrix defined by shape may be
    non-contiguous.

    Parameters
    ----------
    m : C-contiguous 2D array
        Array to swap. It may be larger than prod(shape).
    shape : (ndim,) int64 ndarray
        Tensor shape before transpose.
    axes : tuple or 1D array of ndim integers
        Axes permutation.
    n_leg_rows : int
        Number of axes to concatenate to obtain m rows.
    rshift : int
        Shift over m rows.
    cshift : int
        Shift over m columns.

    Returns
    -------
    swapped : 1D array
        Contiguous tensor obtained after swap.
    """
    # indexing with unsigned int should be slightly faster
    # need to wait for numba pull/8333 to be included into release to use unsigned

    rstrides1, rstrides2, rmod, nci = _numba_get_strides(shape, axes, n_leg_rows)
    nrow = shape[:n_leg_rows].prod()
    ncol = shape[n_leg_rows:].prod()
    swapped_in = np.empty((nrow * ncol,), dtype=m.dtype)
    for i in numba.prange(nrow):
        nri = (i // rstrides1 % rmod * rstrides2).sum()
        for j in range(ncol):
            swapped_in[nri + nci[j]] = m[rshift + i, cshift + j]
    return swapped_in


@numba.njit
def _numba_monothread_strided_transpose(m, shape, axes, n_leg_rows, rshift=0, cshift=0):
    """
    Single thread version of _numba_parallel_strided_transpose
    """
    rstrides1, rstrides2, rmod, nci = _numba_get_strides(shape, axes, n_leg_rows)
    nrow = shape[:n_leg_rows].prod()
    ncol = shape[n_leg_rows:].prod()
    swapped_in = np.empty((nrow * ncol,), dtype=m.dtype)
    for i in range(nrow):
        nri = (i // rstrides1 % rmod * rstrides2).sum()
        for j in range(ncol):
            swapped_in[nri + nci[j]] = m[rshift + i, cshift + j]
    return swapped_in
