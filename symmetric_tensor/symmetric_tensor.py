import numpy as np
import scipy.linalg as lg

from misc_tools.svd_tools import sparse_svd, numba_find_chi_largest


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

        # need to take into account possibly missing block in self or other
        blocks = []
        block_irreps = []
        i1, i2 = 0, 0
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_irreps[i1] == other._block_irreps[i2]:
                blocks.append(self._blocks[i1] + other._blocks[i2])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < other._block_irreps[i2]:
                blocks.append(self._blocks[i1].copy())  # no data sharing
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
            else:
                blocks.append(other._blocks[i2].copy())  # no data sharing
                block_irreps.append(other._block_irreps[i2])
                i2 += 1

        blocks = tuple(blocks)
        block_irreps = np.array(block_irreps)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, block_irreps)

    def __sub__(self, other):
        assert type(other) == type(self), "Mixing incompatible types"
        assert self._shape == other._shape, "Mixing incompatible axes"
        assert self._n_row_legs == other._n_row_legs, "Mixing incompatible fusion trees"

        # need to take into account possibly missing block in self or other
        blocks = []
        block_irreps = []
        i1, i2 = 0, 0
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_irreps[i1] == other._block_irreps[i2]:
                blocks.append(self._blocks[i1] - other._blocks[i2])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < other._block_irreps[i2]:
                blocks.append(self._blocks[i1].copy())  # no data sharing
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
            else:
                blocks.append(-other._blocks[i2])
                block_irreps.append(other._block_irreps[i2])
                i2 += 1

        blocks = tuple(blocks)
        block_irreps = np.array(block_irreps)
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
        return self

    def __itruediv__(self, x):
        for b in self._blocks:
            b /= x
        return self

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

    def norm(self):
        """
        Tensor Frobenius norm.
        """
        return NotImplemented

    @property
    def T(self):
        """
        Matrix transpose operation, swapping rows and columns. Internal structure and
        and leg merging are not affected: each block is just transposed. Block irreps
        are conjugate, which may change block order.
        """
        # Transpose the matrix representation of the tensor, ie swap rows and columns
        # and transpose diagonal blocks, without any data move or copy. Irreps need to
        # be conjugate since row (bra) and columns (ket) are swapped. Since irreps are
        # just integers, conjugation is group-specific and cannot be implemented here.
        return NotImplemented

    def conjugate(self):
        """
        Complex conjugate operation. Block values are conjugate, block_irreps are also
        conjugate according to group rules. Internal structure is not affected, however
        block order may change.
        """
        return NotImplemented

    @property
    def H(self):
        """
        Hermitian conjugate operation, swapping rows and columns and conjugating blocks.
        block_irreps and block order are not affected.
        """
        # block_irreps are conjugate both in T and conj: no change
        nlr = self._ndim - self._n_leg_rows
        blocks = tuple(b.T.conj() for b in self._blocks)
        axis_reps = tuple(self._axis_reps[i] for i in range(-nlr, self._n_leg_rows))
        return type(self)(axis_reps, nlr, blocks, self._block_irreps)

    def permutate(self, row_axes, col_axes):  # signature != ndarray.transpose
        """
        Permutate axes, changing tensor structure.
        """
        return NotImplemented

    def __matmul__(self, other):
        """
        Tensor dot operation between two tensors with compatible internal structure.
        Left hand term column axes all are contracted with right hand term row axes.
        """
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

    def svd(self, cut=None, window=0, rcutoff=0.0, degen_ratio=1.0):
        """
        Compute block-wise SVD of self and keep only cut largest singular values. Do not
        truncate if cut is not provided. Keep only values larger than rcutoff * max(sv).
        """
        full = cut is None
        raw_u = [None] * self._nblocks
        raw_s = [None] * self._nblocks
        raw_v = [None] * self._nblocks
        for bi, b in enumerate(self._blocks):
            if full or min(b.shape) < 3 * cut:  # dense svd for small blocks
                try:
                    u, s, v = lg.svd(b, full_matrices=False, check_finite=False)
                except lg.LinAlgError as err:
                    print("Error in scipy dense SVD:", err)
                    u, s, v = lg.svd(
                        b, full_matrices=False, check_finite=False, driver="gesvd"
                    )
            else:
                u, s, v = sparse_svd(b, k=cut + window)
            raw_u[bi] = u
            raw_s[bi] = s
            raw_v[bi] = v

        if full:
            cutoff = rcutoff * max(s[0] for s in raw_s)
            block_cuts = np.array([(s > cutoff).sum() for s in raw_s])
        else:
            raw_s = tuple(raw_s)
            block_cuts = numba_find_chi_largest(raw_s, cut, rcutoff, degen_ratio)

        block_irreps = []
        u_blocks = []
        s_values = []
        v_blocks = []
        for bi, c in enumerate(block_cuts):
            if c:
                block_irreps.append(self._block_irreps[bi])
                u_blocks.append(np.ascontiguousarray(raw_u[bi][:, :c]))
                s_values.append(raw_s[bi][:c])
                v_blocks.append(raw_v[bi][:c])

        block_irreps = np.array(block_irreps)
        mid_rep = self.init_representation(block_cuts, block_irreps)
        rep_u = self._axis_reps[: self._n_leg_rows] + (mid_rep,)
        rep_v = (mid_rep,) + self._axis_reps[self._n_leg_rows :]
        U = type(self)(rep_u, self._n_leg_rows, u_blocks, block_irreps)
        V = type(self)(rep_v, 1, v_blocks, block_irreps)
        return U, s_values, block_irreps, V
