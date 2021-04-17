import numpy as np
import scipy.sparse as ssp


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
