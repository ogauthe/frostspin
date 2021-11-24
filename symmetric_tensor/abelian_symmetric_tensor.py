import numpy as np
import scipy.linalg as lg
import numba

from .symmetric_tensor import SymmetricTensor


@numba.njit(parallel=True)
def _numba_fill_block(m, ri, ci):
    b = np.empty((ri.size, ci.size), dtype=m.dtype)
    for i in numba.prange(ri.size):
        for j in numba.prange(ci.size):
            b[i, j] = m[ri[i], ci[j]]
    return b


@numba.njit
def _numba_reduce_to_blocks(m, row_irreps, col_irreps):
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
            b = _numba_fill_block(m, ri, ci)  # parallel
            blocks.append(b)
            block_irreps.append(sorted_row_irreps[row_blocks[rbi]])
    block_irreps = np.array(block_irreps)
    return blocks, block_irreps


@numba.njit(parallel=True)
def _numba_blocks_to_array(blocks, block_irreps, row_irreps, col_irreps):
    # blocks must be homogeneous C-array tuple
    # heterogeneous tuple fails on __getitem__
    # homogeneous F-array MAY fail in a non-deterministic way
    m = np.zeros((row_irreps.size, col_irreps.size), dtype=blocks[0].dtype)
    for bi in range(len(blocks)):  # same as transpose => no parallel
        row_indices = (row_irreps == block_irreps[bi]).nonzero()[0]
        col_indices = (col_irreps == block_irreps[bi]).nonzero()[0]
        for i in numba.prange(row_indices.size):
            for j in numba.prange(col_indices.size):
                m[row_indices[i], col_indices[j]] = blocks[bi][i, j]
    return m


@numba.njit
def _numba_get_indices(irreps):
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


@numba.njit  # jit to inline in abelian_tranpose
def _numpy_get_indices(irreps):
    perm = irreps.argsort(kind="mergesort")
    sorted_irreps = irreps[perm]
    block_bounds = (
        [0]
        + list((sorted_irreps[:-1] != sorted_irreps[1:]).nonzero()[0] + 1)
        + [irreps.size]
    )
    n = len(block_bounds) - 1
    unique_irreps = np.empty((n,), dtype=np.int8)
    irrep_count = np.empty((n,), dtype=np.int64)
    block_indices = np.empty(irreps.shape, dtype=np.int64)
    irrep_blocks = np.empty(irreps.shape, dtype=np.int64)
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
def _numba_abelian_transpose(
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
    old_blocks : homogeneous tuple of onb C-array
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
    blocks : tuple of nnb C-array
        Reduced blocks after transpose.
    block_irreps : (nnb,) int8 ndarray
        Block irreps after transpose.

    Note that old_shape is a ndarray and not a tuple.
    old_blocks MUST be homogeneous tuple of C-array, using F-array sometimes
    fails in a non-deterministic way.
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
    ) = _numpy_get_indices(new_row_irreps)

    # 3) find each column index inside new blocks
    block_cols = np.empty((new_col_irreps.size,), dtype=np.int64)
    col_irrep_count = np.zeros((unique_row_irreps.size,), dtype=np.int64)
    for i in range(new_col_irreps.size):
        for j in range(unique_row_irreps.size):
            if new_col_irreps[i] == unique_row_irreps[j]:
                block_cols[i] = col_irrep_count[j]
                col_irrep_count[j] += 1
                break

    # 4) initialize block sizes. Non-existing blocks stay zero-sized
    # we need to keep all irrep blocks including empty ones (=no column) so that
    # new_row_block_indices still refers to the accurate block.
    # We need to initialize to zero and not to empty because of possibly missing old
    # block.
    # >> other possibility: contiguous array data of size ncoeff, new_blocks information
    # set with strides. Then new_blocks = [data[i:j].reshape(m,n)]
    dtype = old_blocks[0].dtype
    new_blocks = [
        np.zeros((row_irrep_count[i], col_irrep_count[i]), dtype=dtype)
        for i in range(unique_row_irreps.size)
    ]

    # 5) copy all coeff from all blocks to new destination
    # much faster NOT to parallelize loop on old_blocks (huge difference in block sizes)
    for bi in range(old_block_irreps.size):
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

    # 6) drop empty blocks, we do not need new_row_block_indices anymore
    blocks = []
    block_irreps = []
    for i in range(unique_row_irreps.size):
        if new_blocks[i].size:
            blocks.append(new_blocks[i])
            block_irreps.append(unique_row_irreps[i])

    block_irreps = np.array(block_irreps)
    return blocks, block_irreps


class AbelianSymmetricTensor(SymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with abelian symmetry. Irreps
    are labelled as int8 integers, representations are 1D ndarray with np.int8 dtype.
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    _symmetry = NotImplemented

    @staticmethod
    def init_representation(degen, irreps):
        rep = np.empty((degen.sum(),), dtype=np.int8)
        k = 0
        for (d, irr) in zip(degen, irreps):
            rep[k : k + d] = irr
            k += d
        return rep

    @staticmethod
    def representation_dimension(rep):
        return rep.size

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, conjugate_columns=True):
        assert arr.shape == tuple(r.size for r in row_reps + col_reps)
        if conjugate_columns:
            col_reps = tuple(cls.conjugate_representation(r) for r in col_reps)
        row_irreps = cls.combine_representations(*row_reps)
        col_irreps = cls.combine_representations(*col_reps)
        # requires copy if arr is not contiguous
        # using flatiter on non-contiguous is too slow, no other way
        M = arr.reshape(row_irreps.size, col_irreps.size)
        blocks, block_irreps = _numba_reduce_to_blocks(M, row_irreps, col_irreps)
        assert (
            abs((n := lg.norm(arr)) - np.sqrt(sum(lg.norm(b) ** 2 for b in blocks)))
            <= 1e-13 * n  # allows for arr = 0
        )
        return cls(row_reps, col_reps, blocks, block_irreps)

    def _toarray(self):
        return _numba_blocks_to_array(
            self._blocks,
            self._block_irreps,
            self.get_row_representation(),
            self.get_column_representation(),
        )

    def _permutate(self, row_axes, col_axes):
        # construt new axes, conjugate if axis changes between row adnd column
        nrr = len(self._row_reps)
        row_reps = []
        for ax in row_axes:
            if ax < nrr:
                row_reps.append(self._row_reps[ax])
            else:
                row_reps.append(self.conjugate_representation(self._col_reps[ax - nrr]))
        col_reps = []
        for ax in col_axes:
            if ax < nrr:
                col_reps.append(self.conjugate_representation(self._row_reps[ax]))
            else:
                col_reps.append(self._col_reps[ax - nrr])

        # construct new blocks by swapping coeff
        blocks, block_irreps = _numba_abelian_transpose(
            np.array(self._shape),
            self._blocks,
            self._block_irreps,
            self.get_row_representation(),
            self.get_column_representation(),
            nrr,
            row_axes + col_axes,
            self.combine_representations(*row_reps),
            self.combine_representations(*col_reps),
        )
        return type(self)(row_reps, col_reps, blocks, block_irreps)

    def group_conjugated(self):
        conj_irreps = self.conjugate_representation(self._block_irreps)  # abelian only
        so = conj_irreps.argsort()
        block_irreps = conj_irreps[so]
        blocks = tuple(self._blocks[i] for i in so)
        row_reps = tuple(self.conjugate_representation(r) for r in self._row_reps)
        col_reps = tuple(self.conjugate_representation(r) for r in self._col_reps)
        return type(self)(row_reps, col_reps, blocks, block_irreps)

    def check_blocks_fit_representations(self):
        assert self._block_irreps.size == self._nblocks
        assert len(self._blocks) == self._nblocks
        row_irreps = self.get_row_representation()
        col_irreps = self.get_column_representation()
        for (irr, b) in zip(self._block_irreps, self._blocks):
            nr = (row_irreps == irr).sum()
            nc = (col_irreps == irr).sum()
            assert nr > 0
            assert nc > 0
            assert b.shape == (nr, nc)
        return True

    def norm(self):
        return np.sqrt(sum(lg.norm(b) ** 2 for b in self._blocks))
