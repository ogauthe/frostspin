import numba
import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg


def find_block_cuts(
    block_values, cut_target, *, dims=None, atol=0.0, rtol=None, degen_ratio=1.0
):
    """
    Find chi largest values from a tuple of blockwise, decreasing singular values.
    Assume number of blocks is small: block_max_val is never sorted and elements
    are compared at each iteration.

    Parameters
    ----------
    block_values: enumerable of 1D ndarray
        Sorted values by block. Each block must be real postive  and sorted by
        decreasing values
    cut_target: int
        Number of values to keep. This is a target, the actual value may be bigger to
        preserve multiplets.
    dims: array of integer
        Degeneracy of each block. A given value in block i counts for dims[i] * i to
        reach chi. If None, assumed to be 1 everywhere.
    atol : float, optional
        Absolute threshold term, default value is 0.
    rtol : float, optional
        Relative threshold term, default value is ``sum(length(values)) * eps`` where
        ``eps`` is the machine precision value of the datatype of ``values``.
    degen_ratio: float
        ratio to keep degenerate values. Default to 1.0 (keep values exactly degenerate)

    Returns
    -------
    block_cuts: integer ndarray
        Number of values to keep in each block.
    """
    block_values = tuple(np.asarray(t) for t in block_values)
    dt = block_values[0].dtype
    if dims is None:
        dims = np.ones((len(block_values),), dtype=int)
    else:
        dims = np.asarray(dims, dtype=int, order="C")
    rtol = (
        sum(len(b) for b in block_values) * np.finfo(dt).eps if (rtol is None) else rtol
    )
    cutoff = float(atol + rtol * max([b[0] for b in block_values]))
    degen_ratio = float(degen_ratio)
    assert dims.shape == (len(block_values),)
    assert degen_ratio <= 1.0
    return _numba_find_block_cuts(
        block_values, int(cut_target), dims, cutoff, degen_ratio
    )


@numba.njit
def _numba_find_block_cuts(block_s, cut_target, dims, cutoff, degen_ratio):
    block_max_vals = np.array([b[0] for b in block_s])
    block_cuts = np.zeros((len(block_s),), dtype=np.int64)
    kept = 0
    bi = block_max_vals.argmax()
    while block_max_vals[bi] > cutoff and kept < cut_target:
        c = degen_ratio * block_max_vals[bi]
        while block_max_vals[bi] >= c:  # take all quasi-degenerated values together
            block_cuts[bi] += 1
            kept += dims[bi]
            if block_cuts[bi] < block_s[bi].size:
                block_max_vals[bi] = block_s[bi][block_cuts[bi]]
            else:
                block_max_vals[bi] = -1.0
            bi = block_max_vals.argmax()

    return block_cuts


def robust_eigh(a, *, compute_vectors=True):
    """
    Wrapper for scipy.linalg.eigh. Catch a LinAlgError and retries with another driver.

    Parameters
    ----------
    a : ndarray with shape (n, n)
        Square dense matrix to decompose.
    compute_vectors: bool
        Whether to compute eigenvectors. Default is True.

    Returns
    -------
    s : ndarray with shape (n,)
        The eigenvalues, sorted in non-increasing order.
    u : ndarray with shape (n, n)
        Unitary matrix having normalized eigenvectors as columns.

    For compute_vectors=False, only s is returned. Similarly to scipy, wrong results wil
    be returned if a is not self-adjoint.
    """
    eigvals_only = not compute_vectors
    driver = "evd" if a.size > 1 else "evr"  # scipy issue 20512
    try:
        out = lg.eigh(a, eigvals_only=eigvals_only, driver=driver)
    except lg.LinAlgError as err:
        print(f"Warning: eigh: evd failed with {err}. Try evr")
        try:
            out = lg.eigh(
                a, eigvals_only=eigvals_only, check_finite=False, driver="evr"
            )
        except lg.LinAlgError as err2:
            print(f"Warning: eigh: evr failed with {err2}. Try evx")
            out = lg.eigh(
                a, eigvals_only=eigvals_only, check_finite=False, driver="evx"
            )
    return out


def robust_svd(a, *, compute_vectors=True):
    """
    Wrapper for scipy.linalg.svd. Catch a LinAlgError and retries with another driver.

    Parameters
    ----------
    a : ndarray with shape (m, n)
        Dense matrix to decompose.
    compute_vectors: bool
        Whether to compute singular vectors. Default is True.

    Returns
    -------
    u : ndarray with shape (m, min(m, n))
        Unitary matrix having left singular vectors as columns.
    s : ndarray with shape (min(m,n),)
        The singular values, sorted in non-increasing order.
    v : ndarray with shape (min(m, n), n)
        Unitary matrix having right singular vectors as rows.

    For compute_vectors=False, only s is returned.
    """
    try:
        out = lg.svd(
            a, full_matrices=False, compute_uv=compute_vectors, lapack_driver="gesdd"
        )
    except lg.LinAlgError as err:
        print(f"Warning: svd: gesdd failed with {err}. Try gesvd")
        out = lg.svd(
            a,
            full_matrices=False,
            compute_uv=compute_vectors,
            check_finite=False,
            lapack_driver="gesvd",
        )
    return out


def sparse_svd(A, k, *, ncv=None, tol=0, maxiter=None, compute_vectors=True):
    # use custom svds adapted from scipy to remove small value cutoff
    # remove some other unused features: impose which='LM' and solver='arpack', do not
    # allow fixed v0, always sort values by decreasing order
    # compute either no singular vectors or both U and V
    # ref: https://github.com/scipy/scipy/pull/11829

    n, m = A.shape
    dmin = min(m, n)

    if k < 1 or k >= dmin:
        raise ValueError("k must be between 1 and min(A.shape)")
    if n > m:
        X_dot = A.dot
        XH_dot = A.T.conj().dot
        transpose = False
    else:
        X_dot = A.T.conj().dot
        XH_dot = A.dot
        transpose = True

    def dot_XH_X(x):
        return XH_dot(X_dot(x))

    # lobpcg is not reliable, see scipy issue #10974.
    XH_X = slg.LinearOperator(matvec=dot_XH_X, shape=(dmin, dmin), dtype=A.dtype)
    _, eigvec = slg.eigsh(XH_X, k=k, tol=tol, maxiter=maxiter, ncv=ncv)

    # ensure exact orthogonality
    eigvec, _ = lg.qr(eigvec, overwrite_a=True, mode="economic", check_finite=False)

    # improve stability following https://github.com/scipy/scipy/pull/11829
    # matrices should be small enough to avoid convergence errors in lg.svd
    u = X_dot(eigvec)
    if not compute_vectors:
        return lg.svd(u, compute_uv=False, overwrite_a=True)

    # compute the right singular vectors of X and update the left ones accordingly
    u, s, vh = lg.svd(u, full_matrices=False, overwrite_a=True)
    if transpose:
        u, vh = eigvec @ vh.T.conj(), u.T.conj()
    else:
        vh = vh @ eigvec.T.conj()
    return u, s, vh
