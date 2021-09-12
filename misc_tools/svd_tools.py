import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import numba

from groups.toolsU1 import default_color, svdU1


@numba.njit(cache=True)
def numba_find_chi_largest(block_s, chi, rcutoff=0.0, degen_ratio=1.0):
    """
    Find chi largest values from a tuple of blockwise, decreasing singular values.
    Assume number of blocks is small: block_max_val is never sorted and elements
    are compared at each iteration.

    Parameters
    ----------
    block_s: tuple of 1D ndarray sorted by decreasing values
        Sorted values by block
    chi: int
        number of values to keep
    rcutoff: float
        relative cutoff on small values. Default to 0 (no cutoff)
    degen_ratio: float
        ratio to keep degenerate values. Default to 1.0 (keep values exactly degenerate)

    Returns
    -------
    block_cuts: integer ndarray
        Number of values to keep in each block.

    Note that numba requires block_s to be a tuple, a list is not accepted.
    """
    # numba issue #7394
    block_max_vals = np.array([block_s[bi][0] for bi in range(len(block_s))])
    cutoff = block_max_vals.max() * rcutoff
    block_cuts = np.zeros((len(block_s),), dtype=np.int64)
    for kept in range(chi - 1):
        bi = block_max_vals.argmax()
        if block_max_vals[bi] < cutoff:
            break
        block_cuts[bi] += 1
        if block_cuts[bi] < block_s[bi].size:
            block_max_vals[bi] = block_s[bi][block_cuts[bi]]
        else:
            block_max_vals[bi] = -1.0  # in case cutoff = 0

    # keep last multiplet
    bi = block_max_vals.argmax()
    cutoff = max(cutoff, degen_ratio * block_max_vals[bi])
    while block_max_vals[bi] >= cutoff:
        block_cuts[bi] += 1
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
    if full or min(M.shape) < 3 * cut:  # full allows to use numba while cutting
        U, s, V, colors = svdU1(M, row_colors, col_colors)
    else:
        U, s, V, colors = sparse_svdU1(  # FIXME: interplay svd_truncate / sparse_svdU1
            M, cut + window, row_colors, col_colors, maxiter=maxiter
        )
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

    XH_X = slg.LinearOperator(
        matvec=matvec_XH_X, dtype=A.dtype, matmat=matmat_XH_X, shape=(dmin, dmin)
    )

    try:
        eigvals, eigvec = slg.eigsh(
            XH_X, k=k, tol=tol, maxiter=maxiter, ncv=ncv, which="LM"
        )
    except slg.ArpackNoConvergence as err:
        print("ARPACK did not converge, use LOBPCG", err)
        X = np.random.RandomState(52).randn(dmin, k)
        eigvals, eigvec = slg.lobpcg(XH_X, X, tol=tol, maxiter=maxiter)

    # improve stability following https://github.com/scipy/scipy/pull/11829
    # matrices should be small enough to avoid convergence errors in lg.svd
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
def sparse_svdU1(M, k_block, row_colors, col_colors, maxiter=1000):
    """
    *** SEVERAL PROBLEMS IN THIS FUNCTION. NOT USED CURRENTLY ***

    Compute a given number of singular values and singular vectors in each U(1) symmetry
    sector. If colors are not provided, fall back to standard sparse SVD, with only one
    block.

    Parameters
    ----------
    M : (m,n) ndarray
      Matrix to decompose.
    k_block : integer
      Number of singular vectors to compute for each block.
    row_colors : (m,) integer ndarray
      U(1) quantum numbers of the rows.
    col_colors : (n,) integer ndarray
      U(1) quantum numbers of the columns.
    maxiter : integer
      Maximal number of Arpack iterations. Finite by default allows clean crash
      instead of running forever.

    Returns
    -------
    U :   U : (m,cut) ndarray
      Left singular vectors.
    s : (cut,) ndarray
      Singular values.
    V : (cut,n) right singular vectors
    color : (cut,) ndarray with int8 data type
      U(1) quantum numbers of s.
    """
    raise NotImplementedError
    # FIXME: dealing with cut, blockwise or global?
    # computing k_block values by block without cutting is clumsy.
    # matrix element sorting at the end is inefficient if cut is made later
    # how to deal with multiplets? Should be done in svd_truncate
    # probably need to inline this in svd_truncate and restrain this function to
    # computation of k largest values, staying close to sparse_svd.

    # revert to standard sparse svd if colors are not provided
    if not row_colors.size or not col_colors.size:
        U, s, V = sparse_svd(M, k=k_block, maxiter=maxiter)
        return U, s, V, default_color

    if M.shape != (row_colors.shape, col_colors.shape):
        raise ValueError("Colors do not match M")

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

    # we need to compute k_block singular values in every sector to deal with the worst
    # case: every k_block largest values in the same sector.
    # A fully memory efficient code would only store 2*cut vectors. Here we select the
    # cut largest values selection once all vector are computed, so we need to store
    # sum( min(cut, sector_size) for each sector) different vectors.
    # Dealing with different block numbers in row_blocks and col_blocks here is
    # cumbersome, so we consider the sub-optimal min(cut, sector_rows_number) while
    # columns number may be smaller.
    max_k = np.minimum(k_block, row_blocks[1:] - row_blocks[:-1]).sum()
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
            if min(m.shape) < 3 * k_block:  # use exact svd for small blocks
                u, s, v = lg.svd(m, full_matrices=False, overwrite_a=True)
            else:
                u, s, v = sparse_svd(m, k=k_block, maxiter=maxiter)

            d = min(k_block, s.size)  # may be smaller than k_block
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
