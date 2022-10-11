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


@numba.njit  # jit to inline in abelian_tranpose
def _numpy_get_indices(irreps):
    perm = irreps.argsort(kind="mergesort").view(np.uint64)
    sorted_irreps = irreps[perm]
    block_bounds = (
        [0]
        + list((sorted_irreps[:-1] != sorted_irreps[1:]).nonzero()[0] + 1)
        + [irreps.size]
    )
    n = len(block_bounds) - 1
    unique_irreps = np.empty((n,), dtype=np.int8)
    irrep_count = np.empty((n,), dtype=np.int64)
    block_indices = np.empty(irreps.shape, dtype=np.uint64)
    irrep_blocks = np.empty(irreps.shape, dtype=np.uint64)
    for i in range(n):
        unique_irreps[i] = sorted_irreps[block_bounds[i]]
        irrep_count[i] = block_bounds[i + 1] - block_bounds[i]
        block_indices[block_bounds[i] : block_bounds[i + 1]] = np.arange(irrep_count[i])
        irrep_blocks[block_bounds[i] : block_bounds[i + 1]] = i
    inv_perm = perm.argsort().view(np.uint64)
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
    old_shape : (ndim,) uint64 ndarray
        Tensor shape before transpose.
    old_blocks : tuple of onb C-array
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
    old_blocks must be numba homogeneous tuple of C-array, using F-array sometimes fails
    in a non-deterministic way.
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
    # indexing is slightly faster with unsigned integers
    ndim = len(axes)
    rstrides1 = np.ones((old_n_leg_rows,), dtype=np.uint64)
    rstrides1[1:] = old_shape[old_n_leg_rows - 1 : 0 : -1]
    rstrides1 = rstrides1.cumprod()[::-1].copy()
    rmod = old_shape[:old_n_leg_rows]

    cstrides1 = np.ones((ndim - old_n_leg_rows,), dtype=np.uint64)
    cstrides1[1:] = old_shape[-1:old_n_leg_rows:-1]
    cstrides1 = cstrides1.cumprod()[::-1].copy()
    cmod = old_shape[old_n_leg_rows:]

    new_strides = np.ones((ndim,), dtype=np.uint64)
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
    ncs = np.uint64(new_col_irreps.size)
    block_cols = np.empty((new_col_irreps.size,), dtype=np.uint64)
    col_irrep_count = np.zeros((unique_row_irreps.size,), dtype=np.uint64)
    for i in range(ncs):
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
        # nonzero returns int64, but unsigned int is slightly faster for indexing
        # due to numba bug, crash if reshape before view, see numba issue 3729
        ori = (old_row_irreps == old_block_irreps[bi]).nonzero()[0].view(np.uint64)
        ori = (ori.reshape(-1, 1) // rstrides1 % rmod * rstrides2).sum(axis=1)
        oci = (old_col_irreps == old_block_irreps[bi]).nonzero()[0].view(np.uint64)
        oci = (oci.reshape(-1, 1) // cstrides1 % cmod * cstrides2).sum(axis=1)
        # ori and oci cannot be empty since old irrep block exists
        for i in numba.prange(ori.size):
            for j in numba.prange(oci.size):
                # nr and nc depend on both ori[i] and oci[j], but they appear several
                # times, in this block as well as in others.
                nr, nc = divmod(ori[i] + oci[j], ncs)
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

    return blocks, block_irreps


class AbelianSymmetricTensor(SymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with abelian symmetry. Irreps
    are labelled as int8 integers, representations are 1D ndarray with np.int8 dtype.
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    @classmethod
    @property
    def singlet(cls):
        return np.zeros((1,), dtype=np.int8)

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

    @staticmethod
    def irrep_dimension(rep):
        return 1

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        assert arr.shape == tuple(r.size for r in row_reps + col_reps)
        nrr = len(row_reps)
        if signature is None:
            signature = np.arange(nrr + len(col_reps)) >= nrr
        else:
            signature = np.ascontiguousarray(signature, dtype=bool)
            assert signature.shape == (nrr + len(col_reps),)

        row_irreps = cls.combine_representations(row_reps, signature[:nrr])
        col_irreps = cls.combine_representations(col_reps, ~signature[nrr:])
        # requires copy if arr is not contiguous
        # using flatiter on non-contiguous is too slow, no other way
        M = arr.reshape(row_irreps.size, col_irreps.size)
        blocks, block_irreps = _numba_reduce_to_blocks(M, row_irreps, col_irreps)
        assert (
            abs((n := lg.norm(arr)) - np.sqrt(sum(lg.norm(b) ** 2 for b in blocks)))
            <= 1e-13 * n  # allows for arr = 0
        ), "norm is not conserved in AbelianSymmetricTensor cast"
        return cls(row_reps, col_reps, blocks, block_irreps, signature)

    def toarray(self, as_matrix=False):
        # cast blocks to C-contiguous to avoid numba bug
        self._blocks = tuple(np.ascontiguousarray(b) for b in self._blocks)
        m = _numba_blocks_to_array(
            self._blocks,
            self._block_irreps,
            self.get_row_representation(),
            self.get_column_representation(),
        )
        if as_matrix:
            return m
        return m.reshape(self._shape)

    @property
    def T(self):
        # in the abelian case, matrix transpose can be obtained by conjugating
        # block_irreps transposing, transpose all blocks and reorder them according to
        # their new (conjugate) irrep.
        conj_irreps = self.conjugate_representation(self._block_irreps)
        so = conj_irreps.argsort()
        block_irreps = conj_irreps[so]
        blocks = tuple(self._blocks[i].T for i in so)
        s = self._signature[np.arange(-self._ndim + self._nrr, self._nrr) % self._ndim]
        return type(self)(self._col_reps, self._row_reps, blocks, block_irreps, s)

    def permutate(self, row_axes, col_axes):
        assert sorted(row_axes + col_axes) == list(range(self._ndim))

        # return early for identity or matrix transpose
        if row_axes == tuple(range(self._nrr)) and col_axes == tuple(
            range(self._nrr, self._ndim)
        ):
            return self
        if row_axes == tuple(range(self._nrr, self._ndim)) and col_axes == tuple(
            range(self._nrr)
        ):
            return self.T

        # avoid numba issue: blocks need to be C-contiguous
        if not all(b.flags["C"] for b in self._blocks):
            if all(b.flags["F"] for b in self._blocks):  # .T returns C-contiguous
                row_axes_T = tuple((ax - self._nrr) % self._ndim for ax in row_axes)
                col_axes_T = tuple((ax - self._nrr) % self._ndim for ax in col_axes)
                return self.T.permutate(row_axes_T, col_axes_T)
            self._blocks = tuple(np.ascontiguousarray(b) for b in self._blocks)

        # construct new row and column representations
        axes = row_axes + col_axes
        nrr = len(row_axes)
        signature = []
        reps = []
        for ax in axes:
            signature.append(self._signature[ax])
            if ax < self._nrr:
                reps.append(self._row_reps[ax])
            else:
                reps.append(self._col_reps[ax - self._nrr])
        signature = np.array(signature)
        row_irreps = self.combine_representations(reps[:nrr], signature[:nrr])
        col_irreps = self.combine_representations(reps[nrr:], ~signature[nrr:])

        # construct new blocks by swapping coeff
        blocks, block_irreps = _numba_abelian_transpose(
            np.array(self._shape, dtype=np.uint64),
            self._blocks,
            self._block_irreps,
            self.get_row_representation(),
            self.get_column_representation(),
            self._nrr,
            axes,
            row_irreps,
            col_irreps,
        )

        tp = type(self)(reps[:nrr], reps[nrr:], blocks, block_irreps, signature)
        assert abs(self.norm() - tp.norm()) <= 1e-13 * self.norm()
        return tp

    def group_conjugated(self):
        conj_irreps = self.conjugate_representation(self._block_irreps)  # abelian only
        so = conj_irreps.argsort()
        block_irreps = conj_irreps[so]
        blocks = tuple(self._blocks[i] for i in so)
        signature = ~self._signature
        return type(self)(
            self._row_reps, self._col_reps, blocks, block_irreps, signature
        )

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

    def toabelian(self):
        return self

    def update_signature(self, sign_update):
        # in the abelian case, bending an index to the left or to the right makes no
        # difference, signs can be ignored.
        up = np.asarray(sign_update, dtype=bool)
        assert up.shape == (self._ndim,)
        row_reps = list(self._row_reps)
        col_reps = list(self._col_reps)
        for i in up.nonzero()[0]:
            if i < self._nrr:
                row_reps[i] = self.conjugate_representation(row_reps[i])
            else:
                j = i - self._nrr
                col_reps[j] = self.conjugate_representation(col_reps[j])
        self._row_reps = tuple(row_reps)
        self._col_reps = tuple(col_reps)
        self._signature = self._signature ^ up
        assert self.check_blocks_fit_representations()
