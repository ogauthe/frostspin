import numba
import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg


def find_chi_largest(block_s, chi, *, dims=None, rcutoff=0.0, degen_ratio=1.0):
    """
    Find chi largest values from a tuple of blockwise, decreasing singular values.
    Assume number of blocks is small: block_max_val is never sorted and elements
    are compared at each iteration.

    Parameters
    ----------
    block_s: enumerable of 1D ndarray sorted by decreasing values
        Sorted values by block.
    chi: int
        Number of values to keep. This is a target, the actual value may be bigger to
        preserve multiplets.
    dims: array of integer
        Degeneracy of each block. A given value in block i counts for dims[i] * i to
        reach chi. If None, assumed to be 1 everywhere.
    rcutoff: float
        relative cutoff on small values. Default to 0 (no cutoff)
    degen_ratio: float
        ratio to keep degenerate values. Default to 1.0 (keep values exactly degenerate)

    Returns
    -------
    block_cuts: integer ndarray
        Number of values to keep in each block.
    """
    block_s = tuple(block_s)
    chi = int(chi)
    if dims is None:
        dims = np.ones((len(block_s),), dtype=int)
    dims = np.asarray(dims, dtype=int, order="C")
    rcutoff = float(rcutoff)
    degen_ratio = float(degen_ratio)
    assert dims.shape == (len(block_s),)
    assert degen_ratio <= 1.0
    block_cuts = _numba_find_chi_largest(block_s, chi, dims, rcutoff, degen_ratio)
    return block_cuts


@numba.njit
def _numba_find_chi_largest(block_s, chi, dims, rcutoff, degen_ratio):
    block_max_vals = np.array([b[0] for b in block_s])
    cutoff = block_max_vals.max() * rcutoff
    block_cuts = np.zeros((len(block_s),), dtype=np.int64)
    kept = 0
    bi = block_max_vals.argmax()
    while block_max_vals[bi] > cutoff and kept < chi:
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
        print(f"Warning: eigh failed with evd, try evr. Error: {err}")
        try:
            out = lg.eigh(
                a, eigvals_only=eigvals_only, check_finite=False, driver="evr"
            )
        except lg.LinAlgError as err2:
            print(f"Warning: eigh failed with evr, try evx. Error: {err2}")
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
        print(f"Warning: svd failed with gesdd, try gesvd. Error: {err}")
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
        s = lg.svd(u, compute_uv=False, overwrite_a=True)
        return s

    # compute the right singular vectors of X and update the left ones accordingly
    u, s, vh = lg.svd(u, full_matrices=False, overwrite_a=True)
    if transpose:
        u, vh = eigvec @ vh.T.conj(), u.T.conj()
    else:
        vh = vh @ eigvec.T.conj()
    return u, s, vh
