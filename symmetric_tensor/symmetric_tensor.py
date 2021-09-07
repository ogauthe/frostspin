import numpy as np
import scipy.linalg as lg
import numba
from numba import literal_unroll  # numba issue #5344

from misc_tools.svd_tools import sparse_svd


# choices are made to make code light and fast:
# irreps are labelled by integers (non-simple groups need another class using lexsort)
# axes are grouped into either row or colum axes to define block-diagonal matrices
# coefficients are stored as dense blocks corresponding to irreps
# blocks are always sorted according to irreps
# blocks are tuple of contiguous F or C arrays (numba)


class SymmetricTensor(object):
    """
    Generic base class to deal with symmetric tensors. Defines interface that can be
    implemented as derived classes for any symmetry.

    Tensors are seen as matrices, with legs grouped into two packs, rows and columns.
    Some authorized blocks are allowed to be missing if they are 0. Such cases arises
    from a matmul call when irrep X appears on left rows and right columns but not in
    the middle representation.
    """

    _symmetry = NotImplemented

    @property
    def symmetry(self):
        return self._symmetry

    # need to define those methods to deal with symmetries
    @classmethod
    def combine_representations(cls, *reps):
        return NotImplemented

    @classmethod
    def conjugate_representation(cls, rep):
        return NotImplemented

    @classmethod
    def init_representation(cls, degen, irreps):
        return NotImplemented

    @classmethod
    def representation_dimension(cls, rep):
        return NotImplemented

    def __init__(self, axis_reps, n_leg_rows, blocks, block_irreps):
        self._axis_reps = axis_reps  # tuple of representation
        self._n_leg_rows = n_leg_rows  # int
        self._shape = tuple(self.representation_dimension(rep) for rep in axis_reps)
        self._ndim = len(axis_reps)
        self._nblocks = len(blocks)
        self._blocks = tuple(blocks)
        self._block_irreps = block_irreps
        self._ncoeff = sum(b.size for b in blocks)
        assert self._nblocks > 0
        assert 0 < n_leg_rows < self._ndim
        assert len(block_irreps) == self._nblocks
        assert sorted(block_irreps) == list(block_irreps)

    @property
    def nblocks(self):
        return self._nblocks

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

    @property
    def matrix_shape(self):
        return (
            np.prod(self._shape[: self._n_leg_rows]),
            np.prod(self._shape[self._n_leg_rows :]),
        )

    @property
    def dtype(self):
        return self._blocks[0].dtype

    @property
    def n_leg_rows(self):
        return self._n_leg_rows

    @property
    def ncoeff(self):
        return self._ncoeff

    @property
    def blocks(self):
        return self._blocks

    @property
    def block_irreps(self):
        return self._block_irreps

    @property
    def axis_reps(self):
        return self._axis_reps

    def copy(self):
        blocks = tuple(b.copy() for b in self._blocks)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def __add__(self, other):
        assert type(other) == type(self), "Mixing incompatible types"
        assert self._shape == other._shape, "Mixing incompatible axes"
        assert self._n_row_legs == other._n_row_legs, "Mixing incompatible fusion trees"
        blocks = tuple(b1 + b2 for (b1, b2) in zip(self._blocks, other._blocks))
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def __sub__(self, other):
        assert type(other) == type(self), "Mixing incompatible types"
        assert self._shape == other._shape, "Mixing incompatible axes"
        assert self._n_row_legs == other._n_row_legs, "Mixing incompatible fusion trees"
        blocks = tuple(b1 - b2 for (b1, b2) in zip(self._blocks, other._blocks))
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def __mul__(self, x):
        assert np.isscalar(x) or x.size == 1
        blocks = tuple(x * b for b in self._blocks)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

    def __neg__(self):
        blocks = tuple(-b for b in self._blocks)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def __imul__(self, x):
        for b in self._blocks:
            b *= x

    def __itruediv__(self, x):
        for b in self._blocks:
            b /= x

    def get_row_representation(self):
        return self.combine_representations(*self._axis_reps[: self._n_leg_rows])

    def get_column_representation(self):
        return self.combine_representations(*self._axis_reps[self._n_leg_rows :])

    def is_heterogeneous(self):
        # blocks may be a numba heterogeneous tuple because a size 1 matrix stays
        # C-contiguous after tranpose and will be cast to numba array(float64, 2d, C),
        # while any larger matrix will be cast to array(float64, 2d, F).
        # see https://github.com/numba/numba/issues/5967
        c_contiguous = [b.flags.c_contiguous for b in self._blocks]
        return min(c_contiguous) ^ max(c_contiguous)

    # symmetry-specific methods with fixed signature
    def toarray(self):
        return NotImplemented

    @property
    def T(self):
        # Transpose the matrix representation of the tensor, ie swap rows and columns
        # and transpose diagonal blocks, without any data move or copy. Irreps need to
        # be conjugate since row (bra) and columns (ket) are swapped. Since irreps are
        # just integers, conjugation is not possible outside of Representation type.
        # This operation is therefore group-specific and cannot be implemented here.
        return NotImplemented

    @property
    def H(self):
        # block_irreps are conjugate both in T and conj: no change
        n_legs = self._ndim - self._n_leg_rows
        blocks = tuple(b.T.conj() for b in self._blocks)
        axis_reps = (
            self._axis_reps[self._n_leg_rows :] + self._axis_reps[: self._n_leg_rows]
        )
        return type(self)(axis_reps, n_legs, blocks, self._block_irreps)

    def permutate(self, row_axes, col_axes):  # signature != ndarray.transpose
        return NotImplemented

    def conjugate(self):
        return NotImplemented

    def __matmul__(self, other):
        # do not construct empty blocks: those will be missing TODO: change this
        assert self._shape[self._n_leg_rows :] == other._shape[: other._n_leg_rows]
        assert all(
            np.asanyarray(r1 == r2).all()
            for (r1, r2) in zip(self._axis_reps[self._n_leg_rows :], other._axis_reps)
        )

        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_irreps[i1] == other._block_irreps[i2]:
                blocks.append(self._blocks[i1] @ other._blocks[i2])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < other._block_irreps[i2]:
                i1 += 1
            else:
                i2 += 1

        block_irreps = np.array(block_irreps)
        axis_reps = (
            self._axis_reps[: self._n_leg_rows] + other._axis_reps[other._n_leg_rows :]
        )
        return type(self)(axis_reps, self._n_leg_rows, blocks, block_irreps)

    def svd(self, cut=np.inf, rcutoff=0.0):
        """
        Compute block-wise SVD of self and keep only cut largest singular values. Do not
        truncate if cut is not provided. Keep only values larger than rcutoff * max(sv).
        """
        # TODO: use find_chi_largest from master
        block_u = [None] * self._nblocks
        block_s = [None] * self._nblocks
        block_v = [None] * self._nblocks
        block_max_vals = np.empty(self._nblocks)
        for bi, b in enumerate(self._blocks):
            if min(b.shape) < 3 * cut:  # dense svd for small blocks
                try:
                    block_u[bi], block_s[bi], block_v[bi] = lg.svd(
                        b, full_matrices=False, check_finite=False
                    )
                except lg.LinAlgError as err:
                    print("Error in scipy dense SVD:", err)
                    block_u[bi], block_s[bi], block_v[bi] = lg.svd(
                        b, full_matrices=False, check_finite=False, driver="gesvd"
                    )
            else:
                block_u[bi], block_s[bi], block_v[bi] = sparse_svd(b, k=cut)
            block_max_vals[bi] = block_s[bi][0]

        cutoff = block_max_vals.max() * rcutoff  # cannot be set before 1st loop
        block_cuts = [0] * self._nblocks
        if cut == np.inf:
            if rcutoff > 0.0:  # remove values smaller than cutoff
                for bi, bs in enumerate(block_s):
                    keep = (bs > cutoff).nonzero()[0]
                    if keep.size:
                        block_cuts[bi] = keep[-1] + 1
            else:
                block_cuts = [b.size for b in block_s]
        else:  # Assume number of blocks is small, block_max_val is never sorted
            k = 0  # and elements are compared at each iteration
            while k < cut:
                bi = block_max_vals.argmax()
                if block_max_vals[bi] < cutoff:
                    break
                block_cuts[bi] += 1
                if block_cuts[bi] < block_s[bi].size:
                    block_max_vals[bi] = block_s[bi][block_cuts[bi]]
                else:
                    block_max_vals[bi] = -1.0  # in case cutoff = 0
                k += 1

        s = []
        mid_irreps = []
        for bi in reversed(range(self._nblocks)):  # reversed to del
            if block_cuts[bi]:
                block_u[bi] = block_u[bi][:, : block_cuts[bi]]
                s.extend(block_s[bi][: block_cuts[bi]][::-1])
                block_v[bi] = block_v[bi][: block_cuts[bi]]
                mid_irreps.append(self._block_irreps[bi])
            else:  # do not keep empty matrices
                del block_u[bi]
                del block_v[bi]

        mid_irreps = tuple(mid_irreps[::-1])
        mid_rep = self._symmetry(block_cuts, mid_irreps)
        rep_u = self._axis_reps[: self._n_leg_rows] + (mid_rep,)
        rep_v = (mid_rep,) + self._axis_reps[self._n_leg_rows :]

        U = type(self)(rep_u, self._n_leg_rows, block_u, mid_rep.irreps)
        V = type(self)(rep_v, 1, block_v, mid_rep.irreps)
        s = np.array(s[::-1])  # cancel reversed in truncation loop
        return U, s, V


class AsymmetricTensor(SymmetricTensor):
    """
    Tensor with no symmetry, mostly for debug and benchmarks purposes.
    """

    # not a subclass of AbelianSymmetricTensor
    # representation is just an integer corresponding to the dimension
    _symmetry = "{e}"
    _irrep = np.zeros((1,), dtype=np.int8)

    @classmethod
    def combine_representations(cls, *reps):
        return np.prod([r for r in reps])

    @classmethod
    def conjugate_representation(cls, rep):
        return rep

    @classmethod
    def init_representation(cls, degen, irreps):
        return degen

    @classmethod
    def representation_dimension(cls, rep):
        return rep

    @classmethod
    def from_array(cls, arr, n_leg_rows):
        matrix_shape = (
            np.prod(arr.shape[:n_leg_rows]),
            np.prod(arr.shape[n_leg_rows:]),
        )
        block = arr.reshape(matrix_shape)
        return cls(arr.shape, n_leg_rows, (block,), cls._irrep)

    def toarray(self):
        return self._blocks[0].reshape(self._shape)

    def permutate(self, row_axes, col_axes):
        arr = self._blocks[0].reshape(self._shape).transpose(row_axes + col_axes)
        return AsymmetricTensor.from_array(arr, len(row_axes))

    @property
    def T(self):
        return AsymmetricTensor(
            self._axis_reps[self._n_leg_rows :] + self._axis_reps[: self._n_leg_rows],
            self._ndim - self._n_leg_rows,
            (self._blocks[0].T,),
            self._irrep,
        )

    def conjugate(self):
        blocks = (self._blocks[0].conj(),)
        return AsymmetricTensor(self._axis_reps, self._n_leg_rows, blocks, self._irrep)

    def norm(self):
        return lg.norm(self._blocks[0])


@numba.njit(parallel=True)
def fill_block(M, ri, ci):
    m = np.empty((ri.size, ci.size), dtype=M.dtype)
    for i in numba.prange(ri.size):
        for j in numba.prange(ci.size):
            m[i, j] = M[ri[i], ci[j]]
    return m


@numba.njit
def numba_reduce_to_blocks(M, row_irreps, col_irreps):
    row_sort = row_irreps.argsort(kind="mergesort")
    sorted_row_irreps = row_irreps[row_sort]
    row_blocks = (
        [0]
        + list((sorted_row_irreps[:-1] != sorted_row_irreps[1:]).nonzero()[0] + 1)
        + [row_irreps.size]
    )
    blocks = []
    block_irreps = []
    for rbi in range(len(row_blocks) - 1):  # order matters: cannot parallelize
        # actually parallelization is possible: init blocks and block_irreps with size
        # len(row_blocks) - 1, parallelize loop and drop empty blocks at the end
        # currently this function is too fast for any reliable benchmark => irrelevant
        ci = (col_irreps == sorted_row_irreps[row_blocks[rbi]]).nonzero()[0]
        if ci.size:
            ri = row_sort[row_blocks[rbi] : row_blocks[rbi + 1]]
            m = fill_block(M, ri, ci)  # parallel
            blocks.append(m)
            block_irreps.append(sorted_row_irreps[row_blocks[rbi]])
    block_irreps = np.array(block_irreps)
    return blocks, block_irreps


@numba.njit
def heterogeneous_blocks_to_array(M, blocks, block_irreps, row_irreps, col_irreps):
    # tedious dealing with heterogeneous tuple: cannot parallelize, enum or getitem
    bi = 0
    for b in literal_unroll(blocks):
        row_indices = (row_irreps == block_irreps[bi]).nonzero()[0]
        col_indices = (col_irreps == block_irreps[bi]).nonzero()[0]
        for i, ri in enumerate(row_indices):
            for j, cj in enumerate(col_indices):
                M[ri, cj] = b[i, j]
        bi += 1


@numba.njit(parallel=True)
def homogeneous_blocks_to_array(M, blocks, block_irreps, row_irreps, col_irreps):
    # when blocks is homogeneous, loops are simple and can be parallelized
    for bi in numba.prange(len(blocks)):
        row_indices = (row_irreps == block_irreps[bi]).nonzero()[0]
        col_indices = (col_irreps == block_irreps[bi]).nonzero()[0]
        for i in numba.prange(row_indices.size):
            for j in numba.prange(col_indices.size):
                M[row_indices[i], col_indices[j]] = blocks[bi][i, j]


@numba.njit(parallel=True)
def fill_transpose(aflat, row_indices, col_indices):
    m = np.empty((row_indices.size, col_indices.size), dtype=aflat.dtype)
    for i in numba.prange(row_indices.size):
        for j in numba.prange(col_indices.size):
            m[i, j] = aflat[row_indices[i] + col_indices[j]]
    return m


@numba.njit
def numba_get_indices(irreps):
    unique_irreps = [irreps[0]]  # crash if irreps.size = 0. Should not happen.
    irrep_count = [0]
    block_indices = np.empty(irreps.shape, dtype=np.int64)
    irrep_blocks = np.empty(irreps.shape, dtype=np.int64)
    for i in range(irreps.size):
        try:
            ind = unique_irreps.index(irreps[i])
        except Exception:  # numba exception matching is limited to <class 'Exception'>
            ind = len(unique_irreps)
            unique_irreps.append(irreps[i])
            irrep_count.append(0)
        block_indices[i] = irrep_count[ind]
        irrep_count[ind] += 1
        irrep_blocks[i] = ind
    unique_irreps = np.array(unique_irreps)
    irrep_count = np.array(irrep_count)
    # unique_irreps is NOT sorted to avoid a 2nd loop on irrep_blocks
    return unique_irreps, irrep_count, block_indices, irrep_blocks


def get_indices(irreps):
    perm = irreps.argsort(kind="stable")
    sorted_irreps = irreps[perm]
    block_bounds = [
        0,
        *((sorted_irreps[:-1] != sorted_irreps[1:]).nonzero()[0] + 1),
        irreps.size,
    ]
    n = len(block_bounds) - 1
    unique_irreps = np.empty((n,), dtype=np.int8)
    irrep_count = np.empty((n,), dtype=int)
    block_indices = np.empty(irreps.shape, dtype=int)
    irrep_blocks = np.empty(irreps.shape, dtype=int)
    for i in range(n):
        unique_irreps[i] = sorted_irreps[block_bounds[i]]
        irrep_count[i] = block_bounds[i + 1] - block_bounds[i]
        block_indices[block_bounds[i] : block_bounds[i + 1]] = np.arange(irrep_count[i])
        irrep_blocks[block_bounds[i] : block_bounds[i + 1]] = i
    inv_perm = perm.argsort()
    block_indices = block_indices[inv_perm]
    irrep_blocks = irrep_blocks[inv_perm]
    return unique_irreps, irrep_count, block_indices, irrep_blocks


@numba.njit(parallel=True)
def numba_abelian_transpose(
    old_shape,
    old_blocks,
    old_block_irreps,
    old_row_irreps,
    old_col_irreps,
    old_n_leg_rows,
    axes,
    new_row_irreps,
    new_col_irreps,
):
    """
    Construct new irrep blocks after permutation.

    Parameters
    ----------
    old_shape : (ndim,) integer ndarray
        Tensor shape before transpose.
    old_blocks : tuple of onb matrices
        Reduced blocks before transpose.
    old_block_irreps : (onb,) int8 ndarray
        Block irreps before transpose.
    old_row_irreps : (old_nrows,) int8 ndarray
        Row irreps before transpose.
    old_col_irreps : (old_ncols,) int8 ndarray
        Column irreps before transpose.
    old_n_leg_rows : int
        Number of axes to concatenate to obtain old rows.
    axes : tuple of ndim integers
        Axes permutation.
    new_row_irreps : (new_nrows,) int8 ndarray
        Row irreps after transpose.
    new_col_irreps : (new_ncols,) int8 ndarray
        Column irreps after transpose.

    Returns
    -------
    blocks : tuple of nnb matrices
        Reduced blocks after transpose.
    block_irreps : (nnb,) int8 ndarray
        Block irreps after transpose.

    Note that old_shape is a ndarray and not a tuple.
    """
    ###################################################################################
    # Loop on old blocks, for each coeff, find new index, new irrep block and copy data.
    #
    # To do this, reduce each index to a flat index and use strides to obtain new flat
    # index, then new row and new column. From new row, get new irrep block, then index
    # of new row and new column in irrep block is needed.
    #
    # What we actually need is:
    # new_row_block_indices: array of indices, size new_row_irreps.size
    #     index of new irrep block of each new row.
    # block_rows: array of index, size new_row_irreps.size
    #     index of each new row within its irrep block
    # block_cols: array of index, size new_col_irreps.size
    #     index of each new col within its irrep block
    # new_blocks: list of arrays with accurate shape
    #
    # it would be pretty similar to loop on new blocks and indices, get old indices and
    # copy old value, but it requires all old blocks to exist. This may not be true with
    # current matmul implementation.
    ###################################################################################

    # 1) construct strides before and after transpose for rows and cols
    # things are much simpler with old_shape as np.array
    ndim = old_shape.size
    rstrides1 = np.ones((old_n_leg_rows,), dtype=np.int64)
    rstrides1[1:] = old_shape[old_n_leg_rows - 1 : 0 : -1]
    rstrides1 = rstrides1.cumprod()[::-1].copy()
    rmod = old_shape[:old_n_leg_rows]

    cstrides1 = np.ones((ndim - old_n_leg_rows,), dtype=np.int64)
    cstrides1[1:] = old_shape[-1:old_n_leg_rows:-1]
    cstrides1 = cstrides1.cumprod()[::-1].copy()
    cmod = old_shape[old_n_leg_rows:]

    new_strides = np.ones(ndim, dtype=np.int64)
    for i in range(ndim - 1, 0, -1):
        new_strides[axes[i - 1]] = new_strides[axes[i]] * old_shape[axes[i]]
    rstrides2 = new_strides[:old_n_leg_rows]
    cstrides2 = new_strides[old_n_leg_rows:]

    # 2) find unique irreps in rows and relate them to blocks and indices.
    (
        unique_row_irreps,
        row_irrep_count,
        block_rows,
        new_row_block_indices,
    ) = numba_get_indices(new_row_irreps)
    block_cols = np.empty((new_col_irreps.size,), dtype=np.int64)
    new_blocks = [np.zeros((0, 0)) for i in range(unique_row_irreps.size)]

    # 3) initialize block sizes. Non-existing blocks stay zero-sized
    # we need to keep all irrep blocks including empty ones (=no column) so that
    # new_row_block_indices still refers to the accurate block.
    # We need to initialize to zero and not to empty because of possibly missing old
    # block.
    # >> other possibility: contiguous array data of size ncoeff, new_blocks information
    # set with strides. Then new_blocks = [data[i:j].reshape(m,n)]
    for i in numba.prange(unique_row_irreps.size):  # maybe no parallel actually better
        irr_indices = (new_col_irreps == unique_row_irreps[i]).nonzero()[0]
        block_cols[irr_indices] = np.arange(irr_indices.size)
        new_blocks[i] = np.zeros((row_irrep_count[i], irr_indices.size))

    # 4) copy all coeff from all blocks to new destination
    for bi in numba.prange(old_block_irreps.size):
        ori = (old_row_irreps == old_block_irreps[bi]).nonzero()[0].reshape(-1, 1)
        ori = (ori // rstrides1 % rmod * rstrides2).sum(axis=1)
        oci = (old_col_irreps == old_block_irreps[bi]).nonzero()[0].reshape(-1, 1)
        oci = (oci // cstrides1 % cmod * cstrides2).sum(axis=1)
        # ori and oci cannot be empty since old irrep block exists
        for i in numba.prange(ori.size):
            for j in numba.prange(oci.size):
                # nr and nc depend on both ori[i] and oci[j], but they appear several
                # times, in this block as well as in others.
                nr, nc = divmod(ori[i] + oci[j], new_col_irreps.size)
                new_bi = new_row_block_indices[nr]
                new_row_index = block_rows[nr]
                new_col_index = block_cols[nc]
                new_blocks[new_bi][new_row_index, new_col_index] = old_blocks[bi][i, j]

    # 5) drop empty blocks, we do not need new_row_block_indices anymore
    blocks = []
    block_irreps = []
    for i in unique_row_irreps.argsort():  # numba_get_indices returns unsorted
        if new_blocks[i].size:
            blocks.append(new_blocks[i])
            block_irreps.append(unique_row_irreps[i])

    block_irreps = np.array(block_irreps)
    return blocks, block_irreps


class AbelianSymmetricTensor(SymmetricTensor):
    # representation is a np.int8 1D array (may change for product groups)
    @classmethod
    def representation_dimension(cls, rep):
        return rep.shape[0]

    @classmethod
    def init_representation(cls, degen, irreps):
        rep = np.empty((degen.sum(),), dtype=np.int8)
        k = 0
        for (d, irr) in zip(degen, irreps):
            rep[k : k + d] = irr
            k += d
        return rep

    @classmethod
    def from_array(cls, arr, axis_reps, n_leg_rows, conjugate_columns=True):
        assert arr.shape == tuple(
            cls.representation_dimension(rep) for rep in axis_reps
        )
        row_axis_reps = axis_reps[:n_leg_rows]
        if conjugate_columns:
            col_axis_reps = tuple(
                cls.conjugate_representation(r) for r in axis_reps[n_leg_rows:]
            )
        else:
            col_axis_reps = axis_reps[n_leg_rows:]
        row_irreps = cls.combine_representations(*row_axis_reps)
        col_irreps = cls.combine_representations(*col_axis_reps)
        # requires copy if arr is not contiguous TODO test and avoid copy if not
        M = arr.reshape(row_irreps.size, col_irreps.size)
        blocks, block_irreps = numba_reduce_to_blocks(M, row_irreps, col_irreps)
        return cls(row_axis_reps + col_axis_reps, n_leg_rows, blocks, block_irreps)

    def toarray(self):
        row_irreps = self.combine_representations(*self._axis_reps[: self._n_leg_rows])
        col_irreps = self.combine_representations(*self._axis_reps[self._n_leg_rows :])
        M = np.zeros(self.matrix_shape, dtype=self.dtype)
        if self.is_heterogeneous():  # TODO treat separatly size 1 + call homogeneous
            heterogeneous_blocks_to_array(
                M, self._blocks, self._block_irreps, row_irreps, col_irreps
            )
        else:
            homogeneous_blocks_to_array(
                M, self._blocks, self._block_irreps, row_irreps, col_irreps
            )
        return M.reshape(self._shape)

    def permutate(self, row_axes, col_axes):
        # it is more convenient to deal woth 1 tuple of axes and use 1 int to
        # split it into rows and columns internally (is it?)
        # but the interface is much simpler with 2 tuples.
        axes = tuple(row_axes) + tuple(col_axes)
        n_leg_rows = len(row_axes)
        t = tuple(range(self._ndim))
        assert sorted(axes) == list(t)
        if n_leg_rows == self._n_leg_rows and axes == t:
            return self

        if (
            n_leg_rows == self._ndim - self._n_leg_rows
            and axes == t[self._n_leg_rows :] + t[: self._n_leg_rows]
        ):
            return self.T

        if self.is_heterogeneous():
            axesT = tuple((ax - self._n_leg_rows) % self._ndim for ax in axes)
            return self.T.permutate(axesT[:n_leg_rows], axesT[n_leg_rows:])

        axis_reps = []
        for i, ax in enumerate(axes):
            if (i < n_leg_rows) ^ (ax < self._n_leg_rows):
                axis_reps.append(self.conjugate_representation(self._axis_reps[ax]))
            else:
                axis_reps.append(self._axis_reps[ax])
        axis_reps = tuple(axis_reps)
        old_row_irreps = self.get_row_representation()
        old_col_irreps = self.get_column_representation()
        new_row_irreps = self.combine_representations(*axis_reps[:n_leg_rows])
        new_col_irreps = self.combine_representations(*axis_reps[n_leg_rows:])
        blocks, block_irreps = numba_abelian_transpose(
            np.array(self._shape),
            self._blocks,
            self._block_irreps,
            old_row_irreps,
            old_col_irreps,
            self._n_leg_rows,
            axes,
            new_row_irreps,
            new_col_irreps,
        )
        return type(self)(axis_reps, n_leg_rows, blocks, block_irreps)

    @property
    def T(self):
        n_legs = self._ndim - self._n_leg_rows
        conj_irreps = self.conjugate_representation(self._block_irreps)  # abelian only
        so = conj_irreps.argsort()
        block_irreps = conj_irreps[so]
        blocks = tuple(self._blocks[i].T for i in so)
        axis_reps = tuple(
            self.conjugate_representation(self._axis_reps[i])
            for i in range(-n_legs, self._n_leg_rows)
        )
        return type(self)(axis_reps, n_legs, blocks, block_irreps)

    def conjugate(self):
        conj_irreps = self.conjugate_representation(self._block_irreps)  # abelian only
        so = conj_irreps.argsort()
        block_irreps = conj_irreps[so]
        blocks = tuple(self._blocks[i].conj() for i in so)
        axis_reps = tuple(self.conjugate_representation(r) for r in self._axis_reps)
        return type(self)(axis_reps, self._n_leg_rows, blocks, block_irreps)

    def norm(self):
        return np.sqrt(sum(lg.norm(b) ** 2 for b in self._blocks))


@numba.njit
def numba_combine_U1(*reps):
    combined = reps[0]
    for r in reps[1:]:
        combined = (combined.reshape(-1, 1) + r).ravel()
    return combined


class U1_SymmetricTensor(AbelianSymmetricTensor):
    _symmetry = "U(1)"

    @classmethod
    def combine_representations(cls, *reps):
        if len(reps) > 1:  # numba issues 7245
            return numba_combine_U1(*reps)
        return reps[0]

    @classmethod
    def conjugate_representation(cls, rep):
        return -rep
