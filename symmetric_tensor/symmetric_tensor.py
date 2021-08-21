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

    def copy(self):
        blocks = tuple(b.copy() for b in self._blocks)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def __add__(self, other):
        assert type(other) == type(self), "Mixing incompatible types"
        assert self._shape == other._shape, "Mixing incompatible axes"
        assert self._n_row_legs == other._n_row_legs, "Mixing incompatible fusion trees"
        blocks = tuple(b1 + b2 for (b1, b2) in zip(self._blocks, other._blocks))
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

    def get_row_representation(self):
        return self.combine_representations(*self._axis_reps[: self._n_leg_rows])

    def get_column_representation(self):
        return self.combine_representations(*self._axis_reps[: self._n_leg_rows])

    def is_heteregeneous(self):
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

    def permutate(self, row_axes, col_axes):  # signature != ndarray.transpose
        return NotImplemented

    def __matmul__(self, other):
        # do not construct empty blocks: those will be missing TODO: change this
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
    col_sort = col_irreps.argsort(kind="mergesort")
    sorted_col_irreps = col_irreps[col_sort]
    row_blocks = (
        [0]
        + list((sorted_row_irreps[:-1] != sorted_row_irreps[1:]).nonzero()[0] + 1)
        + [row_irreps.size]
    )
    col_blocks = (
        [0]
        + list((sorted_col_irreps[:-1] != sorted_col_irreps[1:]).nonzero()[0] + 1)
        + [col_irreps.size]
    )

    blocks = []
    block_irreps = []
    rbi, cbi, rbimax, cbimax = 0, 0, len(row_blocks) - 1, len(col_blocks) - 1
    while rbi < rbimax and cbi < cbimax:
        if sorted_row_irreps[row_blocks[rbi]] == sorted_col_irreps[col_blocks[cbi]]:
            ri = row_sort[row_blocks[rbi] : row_blocks[rbi + 1]]
            ci = col_sort[col_blocks[cbi] : col_blocks[cbi + 1]]
            m = fill_block(M, ri, ci)  # parallel
            blocks.append(m)
            block_irreps.append(sorted_row_irreps[row_blocks[rbi]])
            rbi += 1
            cbi += 1
        elif sorted_row_irreps[row_blocks[rbi]] < sorted_col_irreps[col_blocks[cbi]]:
            rbi += 1
        else:
            cbi += 1
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
        if self.is_heteregeneous():  # TODO treat separatly size 1 + call homogeneous
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
        axes = row_axes + col_axes
        n_leg_rows = len(row_axes)
        assert sorted(axes) == list(range(self.ndim))
        # cast to dense to reshape, transpose to get non-contiguous, then call
        # from_array TODO: from_array currently makes copy
        a = self.toarray().transpose(axes)
        axis_reps = []
        for i, ax in enumerate(axes):
            if (i < n_leg_rows) ^ (ax < self._n_leg_rows):
                axis_reps.append(self.conjugate_representation(self._axis_reps[ax]))
            else:
                axis_reps.append(self._axis_reps[ax])
        axis_reps = tuple(axis_reps)
        return type(self).from_array(a, axis_reps, n_leg_rows, conjugate_columns=False)

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
