import numpy as np
from numba import jit, literal_unroll

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


@jit(nopython=True)
def reduce_matrix_to_blocks(M, row_colors, col_colors):
    """
    Reduce a dense U(1) symmetric matrix into U(1) blocks.

    Parameters
    ----------
    M : (m, n) ndarray
        Matrix to decompose.
    row_colors : (m,) integer ndarray
        U(1) quantum numbers of the rows.
    col_colors : (n,) integer ndarray
        U(1) quantum numbers of the columns.

    Returns
    -------
    block_colors : (k,) list of int
        List of U(1) quantum numbers for every blocks.
    blocks : (k,) list of ndarray
        List of dense blocks with fixed quantum number.
    row_indices : (k,) list of interger 1D arrays
        List of row indices for every blocks.
    col_indices : (k,) list of interger 1D arrays
        List of column indices for every blocks.
    -------
    """
    # quicksort implementation may not be deterministic if a random pivot is used. This
    # is a problem since two matrices with compatible columns and rows may end being
    # incompatible due to different axes permutations in two different calls. A stable
    # sort has a unique solution which solves the problem. It may also allows for more
    # efficient cache use thanks to more contiguous data (and always increasing order)
    row_sort = row_colors.argsort(kind="mergesort")
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

    blocks = []
    block_colors = []
    row_indices = []
    col_indices = []
    rbi, cbi, rbimax, cbimax = 0, 0, len(row_blocks) - 1, len(col_blocks) - 1
    while rbi < rbimax and cbi < cbimax:
        if sorted_row_colors[row_blocks[rbi]] == sorted_col_colors[col_blocks[cbi]]:
            ri = row_sort[row_blocks[rbi] : row_blocks[rbi + 1]].copy()
            ci = col_sort[col_blocks[cbi] : col_blocks[cbi + 1]].copy()
            row_indices.append(ri)  # copy ri to own data and delete row_sort at exit
            col_indices.append(ci)  # same for ci
            m = np.empty((ri.size, ci.size), dtype=M.dtype)
            for i, r in enumerate(ri):
                for j, c in enumerate(ci):
                    m[i, j] = M[r, c]

            blocks.append(m)
            block_colors.append(sorted_row_colors[row_blocks[rbi]])
            rbi += 1
            cbi += 1
        elif sorted_row_colors[row_blocks[rbi]] < sorted_col_colors[col_blocks[cbi]]:
            rbi += 1
        else:
            cbi += 1
    return block_colors, blocks, row_indices, col_indices


@jit(nopython=True)
def blocks_to_array(shape, blocks, row_indices, col_indices):
    ar = np.zeros(shape)
    # blocks may be a numba heterogeneous tuple because a size 1 matrix stays
    # C-contiguous after tranpose and will be cast to numba array(float64, 2d, C), while
    # any larger matrix will be cast to array(float64, 2d, F).
    # cannot enumerate on literal_unroll
    # cannot getitem or zip heterogeneous tuple
    k = 0
    for b in literal_unroll(blocks):
        for i, ri in enumerate(row_indices[k]):
            for j, cj in enumerate(col_indices[k]):
                ar[ri, cj] = b[i, j]
        k += 1
    return ar


class BlockMatrixU1(object):
    """
    Efficient storage for U(1) symmetric matrices.
    """

    def __init__(self, shape, block_colors, blocks, row_indices, col_indices):
        """
        Initialize block matrix. Empty matrices are not allowed.

        Parameters
        ----------
        shape : tuple of two int
            Dense matrix shape.
        block_colors : (k,) list of int
            List of U(1) quantum numbers for every blocks.
        blocks : (k,) list of ndarray
            List of dense blocks with fixed quantum number.
        row_indices : (k,) list of interger 1D arrays
            List of row indices for every blocks.
        col_indices : (k,) list of interger 1D arrays
            List of column indices for every blocks.
        """
        # cast lists to tuple to use them as arguments for numba.
        # do it here to ensure tuple everywhere + numba cannot create tuples.
        assert len(blocks) == len(block_colors) == len(row_indices) == len(col_indices)
        self._nblocks = len(blocks)
        if self._nblocks == 0:
            raise ValueError("At least one nonzero block is required.")
        self._dtype = blocks[0].dtype
        self._shape = shape
        self._block_colors = tuple(block_colors)
        self._blocks = tuple(blocks)
        self._row_indices = tuple(row_indices)
        self._col_indices = tuple(col_indices)

    @classmethod
    def from_dense(cls, M, row_colors, col_colors):
        """
        Create a block matrix from dense matrix M and U(1) quantum numbers for rows
        and columns. row_colors and col_colors have matrix convention, a coefficient at
        position (i, j) is nonzero only if row_colors[i] == col_colors[j], with same
        sign.

        Parameters
        ----------
        M : (m, n) ndarray
            Matrix to decompose.
        row_colors : (m,) integer ndarray
            U(1) quantum numbers of the rows.
        col_colors : (n,) integer ndarray
            U(1) quantum numbers of the columns.

        Returns
        -------
        out : BlockMatrixU1
        """
        assert M.shape == (
            row_colors.size,
            col_colors.size,
        ), "Colors do not match array"
        # put everything inside jitted reduce_matrix_to_blocks function
        (block_colors, blocks, row_indices, col_indices) = reduce_matrix_to_blocks(
            M, row_colors, col_colors
        )
        return cls(M.shape, block_colors, blocks, row_indices, col_indices)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def nblocks(self):
        return self._nblocks

    @property
    def blocks(self):
        return self._blocks

    @property
    def block_colors(self):
        return self._block_colors

    @property
    def row_indices(self):
        return self._row_indices

    @property
    def col_indices(self):
        return self._col_indices

    @property
    def T(self):
        # colors are defined with matrix convention: enters on the row index,
        # c_row[i] - c_col[j] = 0 (while sum(col[axis][coeff]) = 0 for tensors)
        # so need to take opposite of colors.
        # Colors need to stay sorted for __matmul__, so reverse all block-related lists.
        return BlockMatrixU1(
            (self._shape[1], self._shape[0]),
            tuple(-c for c in reversed(self._block_colors)),  # keep sorted colors
            tuple(b.T for b in reversed(self._blocks)),
            self._col_indices[::-1],
            self._row_indices[::-1],
        )

    def copy(self):
        return BlockMatrixU1(
            self._shape,  # tuple of int, no deepcopy needed
            self._block_colors,  # tuple of int
            tuple(b.copy() for b in self._blocks),  # faster than deepcopy
            tuple(ri.copy() for ri in self._row_indices),
            tuple(ci.copy() for ci in self._col_indices),
        )

    def toarray(self):  # numba wrapper
        return blocks_to_array(
            self._shape, self._blocks, self._row_indices, self._col_indices
        )

    def get_block_row_col_with_color(self, color):
        i = self._block_colors.index(color)
        return self._blocks[i], self._row_indices[i], self._col_indices[i]

    def norm(self):
        norm2 = 0.0
        for b in self._blocks:
            norm2 += np.linalg.norm(b) ** 2
        return np.sqrt(norm2)

    def __matmul__(self, other):
        """
        Blockwise matrix product for U(1) symmetric matrices. For a given color, a block
        may exist in one matrix and not in the other, but if both blocks exist they have
        to match.
        """
        # cannot use numba since block may be heterogeneous tuple witout getitem
        i1 = 0
        i2 = 0
        blocks = []
        block_colors = []
        row_indices = []
        col_indices = []
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_colors[i1] == other._block_colors[i2]:
                blocks.append(self._blocks[i1] @ other._blocks[i2])
                block_colors.append(self._block_colors[i1])
                row_indices.append(self._row_indices[i1])
                col_indices.append(other._col_indices[i2])
                i1 += 1
                i2 += 1
            elif self._block_colors[i1] < other._block_colors[i2]:
                i1 += 1
            else:
                i2 += 1
        sh = (self._shape[0], other._shape[1])
        return BlockMatrixU1(sh, block_colors, blocks, row_indices, col_indices)


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
    for c in colors[1:]:
        combined = (combined.reshape(-1, 1) + c).ravel()
    return combined


def diagU1(H, colors):
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
