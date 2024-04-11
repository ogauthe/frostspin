import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import numba


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


def sparse_svd(A, *, k=6, ncv=None, tol=0, maxiter=None, return_singular_vectors=True):
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
    if not return_singular_vectors:
        s = lg.svd(u, compute_uv=False, overwrite_a=True)
        return s

    # compute the right singular vectors of X and update the left ones accordingly
    u, s, vh = lg.svd(u, full_matrices=False, overwrite_a=True)
    if transpose:
        u, vh = eigvec @ vh.T.conj(), u.T.conj()
    else:
        vh = vh @ eigvec.T.conj()
    return u, s, vh
