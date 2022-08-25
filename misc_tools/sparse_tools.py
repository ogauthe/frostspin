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


def sparse_transpose(m, sh1, axes, sh2=None, copy=False, cast="csr"):
    """
    Consider a sparse matrix as a higher rank tensor, transpose its axes and return a
    new sparse matrix.

    Parameters
    ----------
    m : sparse matrix
        Sparse matrix to transpose.
    sh1 : tuple of ints
        Input shape when viewed as a tensor.
    axes : tuple of ints
        New position for the axes after transpose.
    sh2 : tuple of 2 ints
        Output shape. Must be a matrix shape compatible with m. If not provided, output
        shape is the same as input.
    copy : bool
        Whether to copy data. Default is False.
    cast : "coo", "csc" or "csr"
        Sparse matrix ouput format.

    Returns
    -------
    out : csr_matrix
        Transposed tensor cast as a csr_matrix with shape sh2.
    """
    if sorted(axes) != list(range(len(sh1))):
        raise ValueError("axes do not match sh1")
    if sh2 is None:
        sh2 = m.shape
    if len(sh2) != 2:  # csr constructor error is unclear
        raise ValueError("output shape must be a matrix")
    size = m.shape[0] * m.shape[1]
    if np.prod(sh1) != size or np.prod(sh2) != size:
        raise ValueError("invalid matrix shape")

    strides1 = np.array([1, *sh1[:0:-1]]).cumprod()[::-1]
    strides2 = np.array([1, *[sh1[i] for i in axes[:0:-1]]]).cumprod()[::-1]
    ind1D = m.tocoo().reshape(size, 1).row
    ind1D = (ind1D[:, None] // strides1 % sh1)[:, axes] @ strides2
    if cast == "csr":
        return ssp.csr_matrix((m.data, np.divmod(ind1D, sh2[1])), shape=sh2, copy=copy)
    elif cast == "csc":
        return ssp.csc_matrix((m.data, np.divmod(ind1D, sh2[1])), shape=sh2, copy=copy)
    elif cast == "coo":
        return ssp.coo_matrix((m.data, np.divmod(ind1D, sh2[1])), shape=sh2, copy=copy)
    raise ValueError("Unknown sparse matrix format")
