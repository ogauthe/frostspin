import numpy as np
import scipy.linalg as lg
import numba
from numba import literal_unroll  # numba issue #5344

from symmetric_tensor.symmetric_tensor import SymmetricTensor


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
    """
    Efficient storage and manipulation for a tensor with abelian symmetry. Irreps
    are labelled as int8 integers, representations are 1D ndarray with np.int9 dtype.
    """

    @classmethod
    def representation_dimension(cls, rep):
        return rep.size

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
        # requires copy if arr is not contiguous
        # using flatiter on non-contiguous is too slow, no other way
        M = arr.reshape(row_irreps.size, col_irreps.size)
        blocks, block_irreps = numba_reduce_to_blocks(M, row_irreps, col_irreps)
        assert (
            abs(1.0 - np.sqrt(sum(lg.norm(b) ** 2 for b in blocks)) / lg.norm(arr))
            < 1e-13
        )
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
