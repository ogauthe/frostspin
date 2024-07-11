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


def sparse_transpose(m, shape, axes, n_row_axes, *, copy=False):
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
