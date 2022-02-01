import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import numba

from groups.toolsU1 import default_color, svdU1


def find_chi_largest(block_s, chi, dims, rcutoff=0.0, degen_ratio=1.0):
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
        conserve multiplets.
    dims: array of integer
        Dimension of each block. If None, assumed to be 1 everywhere.
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
    dims = np.asarray(dims)
    assert dims.shape == (len(block_s),)
    assert degen_ratio <= 1.0
    return _numba_find_chi_largest(block_s, chi, dims, rcutoff, degen_ratio)


@numba.njit
def _numba_find_chi_largest(block_s, chi, dims, rcutoff, degen_ratio):
    # numba issue #7394
    block_max_vals = np.array([block_s[bi][0] for bi in range(len(block_s))])
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


def svd_truncate(
    M,
    cut,
    row_colors=default_color,
    col_colors=default_color,
    full=False,
    maxiter=1000,
    cutoff=0.0,
    degen_ratio=1.0,
    window=0,
):
    """
    Unique function to compute singular value decomposition of a matrix and truncate.
    If row_colors and col_colors are provided, use U(1) symmetry to compute blockwise
    SVD in every color sectors. Matrices are smaller, but more vectors are computed.

    Parameters
    ----------
    M : (m,n) ndarray
      Matrix to decompose.
    cut : integer
      Number of singular vectors to keep. Actual value depends on kept multiplets.
    row_colors : (m,) integer ndarray
      U(1) quantum numbers of the rows.
    col_colors : (n,) integer ndarray
      U(1) quantum numbers of the columns.
    full : boolean
      Whether to compute the full SVD of the matrix (avoid calling sparse SVD function)
    maxiter : integer
      Maximal number of arpack iterations. Finite by default allows clean crash instead
      of running forever. Not read if full is True.
    cutoff : float
      Singular values smaller than cutoff * max(singular values) are set to zero and
      associated singular vectors are removed. Default is 0.0 (no cutoff)
    degen_ratio : float
        Used to define multiplets in singular values and truncate between two
        multiplets. Two consecutive singular values are considered degenerate if
        1 >= s[i+1]/s[i] >= degen_ratio > 0. Default is 1.0 (exact degeneracies)
    window : integer
      If degen_ratio is not None and full is false, compute cut + window vectors in each
      sector to preserve multiplets. Default is 0.

    Returns
    -------
    U : (m, k) ndarray
      Left singular vectors.
    s : (k,) ndarray, float
      Singular values.
    V : (k, n) ndarray
      Right singular vectors.
    colors : (k,) ndarray, int8
      U(1) quantum numbers of s.

    Notes:
    ------
    Even if a truncature is made, full can be used to compute the full SVD, which may
    be more precise or faster (especially with numba)
    Note that U(1) symmetry forces to compute much more than k vectors, hence a small
    or even 0 window is fine.
    """
    # DO NOT USE
    # keep this function until refactoring simple_update with SymmetricTensor

    if full or min(M.shape) < 3 * cut:  # full allows to use numba while cutting
        U, s, V, colors = svdU1(M, row_colors, col_colors)
    else:  # not used in simple update
        raise NotImplementedError
    # zeros must be removed, without caring for multiplets. If a non-zero multiplet is
    # split, cutoff was not set properly.
    cut = min(cut, (s > cutoff * s[0]).nonzero()[0][-1] + 1)
    nnz = (s[cut:] <= degen_ratio * s[cut - 1 : -1]).nonzero()[0]
    if nnz.size:
        cut += nnz[0]
    U = U[:, :cut]
    s = s[:cut]
    V = V[:cut]
    colors = colors[:cut]
    return U, s, V, colors


def sparse_svd(A, k=6, ncv=None, tol=0, maxiter=None, return_singular_vectors=True):
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
    eigvals, eigvec = slg.eigsh(XH_X, k=k, tol=tol, maxiter=maxiter, ncv=ncv)

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
