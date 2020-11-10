import numpy as np
import scipy.linalg as lg
from scipy.sparse.linalg import eigsh  # use custom svds
from scipy.sparse.linalg.interface import LinearOperator

from toolsU1 import default_color, svdU1


def svd_truncate(
    M,
    cut,
    row_colors=default_color,
    col_colors=default_color,
    full=False,
    maxiter=1000,
    keep_multiplets=False,
    window=10,
    degen_tol=1e-6,
    cutoff=0,
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
      Number of singular vectors to keep. Effective cut depends on keep_multiplets
      (see notes)
    row_colors : (m,) integer ndarray
      U(1) quantum numbers of the rows.
    col_colors : (n,) integer ndarray
      U(1) quantum numbers of the columns.
    full : boolean
      Whether to compute the full SVD of the matrix (avoid calling sparse SVD function
      and preserve numba performances)
    maxiter : integer
      If full is False, maximal number of arpack iterations. Finite by default allows
      clean crash instead of running forever.
    keep_multiplets : bool
      If True, preserve non-abelian symmetry by cutting between two different
      multiplets.
    window : integer
      If keep_multiplets is True and full is false, compute cut + window vectors in each
      sector to preserve multiplets.
    degen_tol : float
      Tolerance to consider two consecutive values as degenerate.
    cutoff : float
      Singular values smaller than cutoff * max(singular values) are set to 0 and
      associated singular vectors are removed.

    Returns
    -------
    U : (m,k) ndarray
      Left singular vectors.
    s : (k,) ndarray, float
      Singular values.
    V : (k,n) ndarray
      Right singular vectors.
    colors : (k,) ndarray, int8
      U(1) quantum numbers of s.

    Notes:
    ------
    cut is fixed if keep_multiplets is False, else as the smallest value in
    [cut, cut + window[ such that s[cut+1] < degen_tol*s[cut].
    Even if a truncature is made, full can be used to compute the full SVD, which may
    be more precise or faster (especially with numba)
    """

    if M.ndim != 2:
        raise ValueError("M has to be a matrix")

    if not row_colors.size or not col_colors.size:  # standard full or sparse SVD
        colors = default_color
        if full or min(M.shape) < 3 * cut:
            U, s, V = lg.svd(M, full_matrices=False)
        else:
            if keep_multiplets:
                U, s, V = sparse_svd(M, k=cut + window, maxiter=maxiter)
            else:
                U, s, V = sparse_svd(M, k=cut, maxiter=maxiter)

    else:  # block SVD using U(1) symmetry
        if full:  # full allows to use numba while cutting
            U, s, V, colors = svdU1(M, row_colors, col_colors)
        else:  # U(1) forces to compute much more than k vectors
            if keep_multiplets:  # small window is fine since ^
                U, s, V, colors = sparse_svdU1(
                    M, cut + window, row_colors, col_colors, maxiter=maxiter
                )
            else:
                U, s, V, colors = sparse_svdU1(
                    M, cut, row_colors, col_colors, maxiter=maxiter
                )

    if cut is not None:
        if keep_multiplets:
            cut += (s[cut:] < degen_tol * s[cut - 1 : -1]).nonzero()[0][0]
        cut = min(cut, (s > cutoff * s[0]).nonzero()[0][-1] + 1)
    else:
        cut = (s > cutoff * s[0]).nonzero()[0][-1] + 1

    # truncate once only with respect to keep_multiplets and zeros_tol
    U = U[:, :cut]
    s = s[:cut]
    V = V[:cut]
    colors = colors[:cut]
    return U, s, V, colors


def sparse_svd(
    A, k=6, ncv=None, tol=0, v0=None, maxiter=None, return_singular_vectors=True
):
    # use custom svds adapted from scipy to remove small value cutoff
    # remove some other unused features: impose which='LM' and solver='arpack'
    # always sort values by decreasing order
    # compute either no singular vectors or both U and V
    # ref: https://github.com/scipy/scipy/pull/11829

    A = np.asarray(A)  # do not consider LinearOperator or sparse matrix
    n, m = A.shape
    dmin = min(m, n)

    if k <= 0 or k >= dmin:
        raise ValueError("k must be between 1 and min(A.shape), k=%d" % k)
    if n > m:
        X_dot = X_matmat = A.dot
        XH_dot = XH_mat = A.T.conj().dot
        transpose = False
    else:
        XH_dot = XH_mat = A.dot
        X_dot = X_matmat = A.T.conj().dot
        transpose = True

    def matvec_XH_X(x):
        return XH_dot(X_dot(x))

    def matmat_XH_X(x):
        return XH_mat(X_matmat(x))

    XH_X = LinearOperator(
        matvec=matvec_XH_X, dtype=A.dtype, matmat=matmat_XH_X, shape=(dmin, dmin)
    )

    eigvals, eigvec = eigsh(
        XH_X, k=k, tol=tol, maxiter=maxiter, ncv=ncv, which="LM", v0=v0
    )
    u = X_matmat(eigvec)
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


# use of eigsh forbids numba
def sparse_svdU1(
    M,
    cut,
    row_colors=default_color,
    col_colors=default_color,
    keep_multiplets=False,
    window=10,
    cuttol=1e-6,
    maxiter=1000,
):
    """
    Compute a given number of singular values and singular vectors using U(1) symmetry.

    Parameters
    ----------
    M : (m,n) ndarray
      Matrix to decompose.
    cut : integer
      Number of singular vectors to compute. Effective cut depends on keep_multiplets
    (see notes)
    row_colors : (m,) integer ndarray
      U(1) quantum numbers of the rows.
    col_colors : (n,) integer ndarray
      U(1) quantum numbers of the columns.
    keep_multiplets : bool
      If true, compute more than cut values and cut between two different multiplets.
    window : integer
      If keep_multiplets is True, compute cut+window vectors and cut between cut and
      cut + window
    cuttol : float
      Tolerance to consider two consecutive values as degenerate.
    maxiter : integer
      Maximal number of iterations. Finite by default allows clean crash instead of
      running forever.

    Returns
    -------
    U :   U : (m,cut) ndarray
      Left singular vectors.
    s : (cut,) ndarray
      Singular values.
    V : (cut,n) right singular vectors
    color : (cut,) ndarray with int8 data type
      U(1) quantum numbers of s.

    Notes:
    ------
    cut is fixed as cut if keep_multiplets is False, else as the smallest value in
    [cut, cut + window[ such that s[cut+1] < cuttol*s[cut].
    It is assumed that degenerate values belong to separate U(1) sectors. This is True
    for SU(2) symmetry, NOT for SU(N>2).
    """

    if row_colors.shape != (M.shape[0],):
        raise ValueError("row_colors has to be (M.shape[0])")
    if col_colors.shape != (M.shape[1],):
        raise ValueError("col_colors has to be (M.shape[1])")

    # optimize cache in matrix block reduction with stable sort
    row_sort = row_colors.argsort(kind="stable")
    sorted_row_colors = row_colors[row_sort]
    col_sort = col_colors.argsort(kind="stable")
    sorted_col_colors = col_colors[col_sort]
    row_blocks = np.array(
        [
            0,
            *((sorted_row_colors[:-1] != sorted_row_colors[1:]).nonzero()[0] + 1),
            M.shape[0],
        ]
    )
    col_blocks = [
        0,
        *((sorted_col_colors[:-1] != sorted_col_colors[1:]).nonzero()[0] + 1),
        M.shape[1],
    ]

    # we need to compute cut singular values in every sector to deal with the worst
    # case: every cut largest values in the same sector.
    # A fully memory efficient code would only store 2*cut vectors. Here we select the
    # cut largest values selection once all vector are computed, so we need to store
    # sum( min(cut, sector_size) for each sector) different vectors.
    # Dealing with different block numbers in row_blocks and col_blocks here is
    # cumbersome, so we consider the sub-optimal min(cut, sector_rows_number) while
    # columns number may be smaller. In CTMRG projector construction, row and column
    # colors are the same anyway.
    max_k = np.minimum(cut, row_blocks[1:] - row_blocks[:-1]).sum()
    U = np.zeros((M.shape[0], max_k))
    S = np.empty(max_k)
    V = np.zeros((max_k, M.shape[1]))
    colors = np.empty(max_k, dtype=np.int8)

    # match blocks with same color and compute SVD inside those blocks only
    k, rbi, cbi, rbimax, cbimax = 0, 0, 0, row_blocks.size - 1, len(col_blocks) - 1
    while rbi < rbimax and cbi < cbimax:
        if sorted_row_colors[row_blocks[rbi]] == sorted_col_colors[col_blocks[cbi]]:
            ri = row_sort[row_blocks[rbi] : row_blocks[rbi + 1]]
            ci = col_sort[col_blocks[cbi] : col_blocks[cbi + 1]]
            m = np.ascontiguousarray(M[ri[:, None], ci])  # no numba
            if min(m.shape) < 3 * cut:  # use exact svd for small blocks
                u, s, v = lg.svd(m, full_matrices=False, overwrite_a=True)
            else:
                u, s, v = sparse_svd(m, k=cut, maxiter=maxiter)

            d = min(cut, s.size)  # may be smaller than cut
            U[ri, k : k + d] = u[:, :d]
            S[k : k + d] = s[:d]
            V[k : k + d, ci] = v[:d]
            colors[k : k + d] = sorted_row_colors[row_blocks[rbi]]
            k += d
            rbi += 1
            cbi += 1
        elif sorted_row_colors[row_blocks[rbi]] < sorted_col_colors[col_blocks[cbi]]:
            rbi += 1
        else:
            cbi += 1

    s_sort = S[:k].argsort()[::-1]  # k <= max_k
    S = S[s_sort]
    U = U[:, s_sort]
    V = V[s_sort]
    colors = colors[s_sort]
    return U, S, V, colors
