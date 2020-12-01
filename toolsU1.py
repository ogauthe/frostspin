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


@jit(nopython=True)
def reduce_matrix_to_blocks(M, row_colors, col_colors):
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
            m = np.empty((ri.size, ci.size))
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

    return blocks, block_colors, row_indices, col_indices


class BlockMatrixU1(object):
    """
    Efficient storage for U(1) symmetric matrices.
    """

    def __init__(self, shape, dtype, blocks, block_colors, row_indices, col_indices):
        assert len(blocks) == len(block_colors) == len(row_indices) == len(col_indices)
        self._shape = shape
        self._dtype = dtype
        self._blocks = blocks
        self._block_colors = block_colors
        self._row_indices = row_indices
        self._col_indices = col_indices
        self._nblocks = len(blocks)

    @classmethod
    def from_dense(cls, M, row_colors, col_colors):
        """
        Constructor from dense matrix M and U(1) quantum numbers for rows and columns.
        """
        assert M.shape == (
            row_colors.size,
            col_colors.size,
        ), "Colors do not match array"
        # put everything inside jitted reduce_matrix_to_blocks function
        (blocks, block_colors, row_indices, col_indices) = reduce_matrix_to_blocks(
            M, row_colors, col_colors
        )
        return cls(M.shape, M.dtype, blocks, block_colors, row_indices, col_indices)

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
        sh = (self._shape[1], self._shape[0])
        blocks = [m.T for m in self._blocks]
        return BlockMatrixU1(
            sh,
            self._dtype,
            blocks,
            self._block_colors,
            self._col_indices,
            self._row_indices,
        )

    def toarray(self):
        ar = np.zeros(self._shape)
        for b, ri, ci in zip(self._blocks, self._row_indices, self._col_indices):
            ar[ri[:, None], ci] = b
        return ar

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
        dtype = max(self._dtype, other._dtype)
        return BlockMatrixU1(sh, dtype, blocks, block_colors, row_indices, col_indices)


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


def dotU1(a, b, rc_a=default_color, cc_a=default_color, cc_b=default_color):
    """
    Optimized matrix product for U(1) symmetric matrices. If colors are not provided,
    revert to standard matmul.

    Parameters
    ----------
    a : (m,l) ndarray
      First argument.
    b : (l,n) ndarray
      Second argument.
    rc_a : (m,) integer ndarray
      U(1) quantum numbers of a rows.
    cc_a : (l,) integer ndarray
      U(1) quantum numbers of a columns. Must be the opposite of b row colors.
    cc_b : (n,) integer ndarray
      U(1) quantum numbers of b columns.

    Returns
    -------
    output : (m,n) ndarray
      dot product of a and b.
    """
    return np.dot(a, b)
    # revert to standard matmul if colors are missing
    if not rc_a.size or not cc_a.size or not cc_b.size:
        # print("dotU1 reverted to np.__matmul__")
        return a @ b

    if a.ndim == b.ndim != 2:
        raise ValueError("ndim must be 2 to use dot")
    if a.shape[0] != rc_a.shape[0]:
        raise ValueError("a rows and row colors shape mismatch")
    if a.shape[1] != cc_a.shape[0]:
        raise ValueError("a columns and column colors shape mismatch")
    if b.shape[1] != cc_b.shape[0]:
        raise ValueError("b columns and column colors shape mismatch")
    if a.shape[1] != b.shape[0]:
        raise ValueError("shape mismatch between a columns and b rows")

    res = np.zeros((a.shape[0], b.shape[1]))
    for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
        ri = (rc_a == -c).nonzero()[0][:, None]
        mid = (cc_a == c).nonzero()[0]
        ci = cc_b == c
        res[ri, ci] = a[ri, mid] @ b[mid[:, None], ci]

    # ex = a @ b
    # r = np.linalg.norm(ex - res) / np.linalg.norm(ex)
    # print("dotU1:\033[33m", r, "\033[0m")
    return res


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


def tensordotU1(a, b, ax_a, ax_b, colors_a=None, colors_b=None):
    """
    Optimized tensor dot product along specified axes for U(1) symmetric tensors.
    If colors are not provided, revert to numpy tensordot.

    Parameters
    ----------
    a,b : ndarray
      tensors to contract.
    ax_a, ax_b : tuple of integers.
      Axes to contract for tensors a and b.
    colors_a, colors_b : list of a.ndim and b.ndim integer arrays.
      U(1) quantum numbers of a and b axes.

    Returns
    -------
    output : ndarray
      Tensor dot product of a and b.
    """
    return np.tensordot(a, b, (ax_a, ax_b))
    # call np.tensordot if colors are not provided
    if (
        colors_a is None
        or colors_b is None
        or not colors_a[0].size
        or not colors_b[0].size
    ):
        # print("tensordotU1 reverted to np.tensordot")
        return np.tensordot(a, b, (ax_a, ax_b))

    if len(ax_a) != len(ax_b):
        raise ValueError("axes for a and b must match")
    if len(ax_a) > a.ndim:
        raise ValueError("axes for a do not match array")
    if len(ax_b) > b.ndim:
        raise ValueError("axes for b do not match array")
    dim_contract = tuple(a.shape[ax] for ax in ax_a)
    if dim_contract != tuple(b.shape[ax] for ax in ax_b):
        raise ValueError("dimensions for a and b do not match")

    # copy np.tensordot
    notin_a = tuple(k for k in range(a.ndim) if k not in ax_a)  # free leg indices
    notin_b = tuple([k for k in range(b.ndim) if k not in ax_b])
    dim_free_a = tuple(a.shape[ax] for ax in notin_a)
    dim_free_b = tuple(b.shape[ax] for ax in notin_b)

    # construct merged colors of a free legs, contracted legs and b free legs
    rc_a = combine_colors(*[colors_a[ax] for ax in notin_a])
    cc_a = combine_colors(*[colors_a[ax] for ax in ax_a])
    cc_b = combine_colors(*[colors_b[ax] for ax in notin_b])

    # np.tensordot algorithm transposes a and b to put contracted axes at end of a and
    # begining of b, reshapes to matrices and compute matrix product.
    # Here avoid complete construction of at and bt which requires copy: compute indices
    # of relevant coeff (depending on colors) of at and bt, then transform them into
    # (flat) indices of a and b. Copy only relevant blocks into small matrices and
    # compute dot product (cannot compute directly indices of a and b because of merged
    # legs)
    sh_at = dim_free_a + dim_contract  # shape of a.transpose(free_a + ax_a)
    prod_a = np.prod(dim_contract)  # offset of free at indices
    div_a = np.array([*sh_at, 1])[:0:-1].cumprod()[::-1]  # 1D index -> multi-index
    # multi-index of at -> 1D index of a by product with transposed shape
    cp_a = np.array([*a.shape, 1])[:0:-1].cumprod()[::-1][np.array(notin_a + ax_a)]

    prod_b = np.prod(dim_free_b)
    sh_bt = dim_contract + dim_free_b
    div_b = np.array([*sh_bt, 1])[:0:-1].cumprod()[::-1]
    cp_b = np.array([*b.shape, 1])[:0:-1].cumprod()[::-1][np.array(ax_b + notin_b)]

    res = np.zeros(dim_free_a + dim_free_b)
    for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
        ri = (rc_a == -c).nonzero()[0][:, None]
        mid = (cc_a == c).nonzero()[0]
        ci = (cc_b == c).nonzero()[0]
        ind_a = ((ri * prod_a + mid)[:, :, None] // div_a % sh_at) @ cp_a
        ind_b = ((mid[:, None] * prod_b + ci)[:, :, None] // div_b % sh_bt) @ cp_b
        res.flat[ri * prod_b + ci] = a.flat[ind_a] @ b.flat[ind_b]

    # ex = np.tensordot(a, b, (ax_a, ax_b))
    # r = np.linalg.norm(ex - res) / np.linalg.norm(ex)
    # print("tensordotU1:\033[33m", r, "\033[0m")
    return res


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
    U = np.zeros((M.shape[0], dmin))
    s = np.empty(dmin)
    V = np.zeros((dmin, M.shape[1]))
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
