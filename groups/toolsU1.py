import numpy as np
from numba import jit

# cannot use None as default color since -default_color or default_color[:l]
# need not to crash.
default_color = np.array([], dtype=np.int8)


def random_U1_tensor(*colors, rng=None):
    """
    Construct random U(1) symmetric tensor. Non-zero coefficients are taken from
    continuous uniform distribution in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    colors : enumerable of 1D integer arrays.
      U(1) quantum numbers of each axis.
    rng : optional, random number generator. Can be used to reproduce results.

    Returns
    -------
    output : float array
      random U(1) tensor, with shape following colors sizes.
    """
    if rng is None:
        rng = np.random.default_rng()
    col1D = combine_colors(*colors)
    nnz = col1D == 0
    t = np.zeros(col1D.size)
    t[nnz] = rng.random(nnz.sum())
    t = t.reshape(tuple(c.size for c in colors))
    return t


def checkU1(T, colorsT, tol=1e-14):
    """
    Check tensor has U(1) symmetry up to tolerance
    """
    if tuple(len(c) for c in colorsT) != T.shape:
        raise ValueError("Color dimensions do not match tensor")
    for ind in np.array((np.abs(T) > tol).nonzero()).T:
        if sum(colorsT[i][c] for i, c in enumerate(ind)) != 0:
            return ind, T[tuple(ind)]
    return None, 0


@jit(nopython=True)
def combine_colors(*colors):
    """
    Construct colors of merged tensor legs from every leg colors.
    """
    if not colors[0].size:
        return default_color
    combined = colors[0]
    if len(colors) > 1:
        for c in colors[1:]:
            combined = (combined.reshape(-1, 1) + c).ravel()
    return combined


def eighU1(H, colors):
    """
    Diagonalize a real symmetric or complex Hermitian U(1) symmetric operator.

    Parameters
    ----------
    H : (M, M) ndarray
      Real symmetric or complex Hermitian U(1) symmetric operator to diagonalize.
    colors : (M,) integer ndarray
      U(1) quantum numbers (same for rows and columns)

    Returns
    -------
    spec : (M,) float ndarray
      Spectrum of H.
    basis : (M, M) ndarray
      Diagonalization basis.
    eigvec_colors : (M,) int8 ndarray
      Colors of H eigenvectors.
    """
    # revert to standard eigh if colors are not provided
    if not colors.size:
        s, U = np.linalg.eigh(H)
        return s, U, default_color

    M = colors.size
    if H.shape != (M, M):
        raise ValueError("Colors do not match H")

    color_sort = colors.argsort(kind="mergesort")
    eigvec_colors = colors[color_sort]
    blocks = [0, *((eigvec_colors[:-1] != eigvec_colors[1:]).nonzero()[0] + 1), M]
    spec = np.empty(M)
    basis = np.zeros((M, M), dtype=H.dtype)

    for i, j in zip(blocks, blocks[1:]):
        m = H[color_sort[i:j, None], color_sort[i:j]]
        spec[i:j], basis[color_sort[i:j], i:j] = np.linalg.eigh(m)

    return spec, basis, eigvec_colors


@jit(nopython=True)
def svdU1(M, row_colors, col_colors):
    """
    Singular value decomposition for a U(1) symmetric matrix M. Revert to standard svd
    if colors are not provided.

    Parameters
    ----------
    M : (m,n) ndarray
      Matrix to decompose.
    row_colors : (m,) integer ndarray
      U(1) quantum numbers of the rows.
    col_colors : (n,) integer ndarray
      U(1) quantum numbers of the columns.

    Returns
    -------
    U : (m,k) ndarray
      Left singular vectors.
    s : (k,) ndarray
      Singular values.
    V : (k,n) right singular vectors
    colors : (k,) integer ndarray
      U(1) quantum numbers of U columns and V rows.

    Note that k may be < min(m,n) if row and column colors do not match on more than
    min(m,n) values. If k = 0 (no matching color), an error is raised to avoid messy
    zero-length arrays (implies M=0, all singular values are 0)
    """
    # revert to standard svd if colors are not provided
    if not row_colors.size or not col_colors.size:
        # print("no color provided, svdU1 reverted to np.linalg.svd")
        U, s, V = np.linalg.svd(M, full_matrices=False)
        return U, s, V, default_color.copy()  # needed for numba

    if M.shape != (row_colors.size, col_colors.size):
        raise ValueError("Colors do not match M")

    row_sort = row_colors.argsort(kind="mergesort")  # optimize block reduction
    sorted_row_colors = row_colors[row_sort]
    col_sort = col_colors.argsort(kind="mergesort")
    sorted_col_colors = col_colors[col_sort]
    row_blocks = (
        [0]
        + list((sorted_row_colors[:-1] != sorted_row_colors[1:]).nonzero()[0] + 1)
        + [M.shape[0]]
    )
    col_blocks = (
        [0]
        + list((sorted_col_colors[:-1] != sorted_col_colors[1:]).nonzero()[0] + 1)
        + [M.shape[1]]
    )
    dmin = min(M.shape)
    U = np.zeros((M.shape[0], dmin), dtype=M.dtype)
    s = np.empty(dmin)
    V = np.zeros((dmin, M.shape[1]), dtype=M.dtype)
    colors = np.empty(dmin, dtype=np.int8)

    # match blocks with same color and compute SVD inside those blocks only
    k, rbi, cbi, rbimax, cbimax = 0, 0, 0, len(row_blocks) - 1, len(col_blocks) - 1
    while rbi < rbimax and cbi < cbimax:
        if sorted_row_colors[row_blocks[rbi]] == sorted_col_colors[col_blocks[cbi]]:
            ri = row_sort[row_blocks[rbi] : row_blocks[rbi + 1]]
            ci = col_sort[col_blocks[cbi] : col_blocks[cbi + 1]]
            m = np.empty((ri.size, ci.size))
            for i, r in enumerate(ri):
                for j, c in enumerate(ci):
                    m[i, j] = M[r, c]
            d = min(m.shape)
            U[ri, k : k + d], s[k : k + d], V[k : k + d, ci] = np.linalg.svd(
                m, full_matrices=False
            )
            colors[k : k + d] = sorted_row_colors[row_blocks[rbi]]
            k += d
            rbi += 1
            cbi += 1
        elif sorted_row_colors[row_blocks[rbi]] < sorted_col_colors[col_blocks[cbi]]:
            rbi += 1
        else:
            cbi += 1

    s_sort = s[:k].argsort()[::-1]
    U = U[:, s_sort]
    s = s[s_sort]
    V = V[s_sort]
    colors = colors[s_sort]

    # r = np.linalg.norm(U * s @ V - M) / np.linalg.norm(M)
    # print("svdU1:\033[33m", r, "\033[0m")
    # if r > 2.0e-14:
    #     raise ValueError("svdU1 failed")
    return U, s, V, colors
