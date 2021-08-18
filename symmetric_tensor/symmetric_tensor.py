import bisect

import numpy as np
import scipy.linalg as lg

from misc_tools.svd_tools import sparse_svd
from groups.abelian_representation import AsymRepresentation, U1_Representation


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

    def __init__(self, axis_reps, n_leg_rows, blocks, block_irreps):
        self._axis_reps = axis_reps  # tuple
        self._n_leg_rows = n_leg_rows  # int
        self._shape = tuple(rep.dim for rep in axis_reps)
        self._ndim = len(axis_reps)
        self._nblocks = len(blocks)
        self._blocks = blocks  # tuple
        self._block_irreps = block_irreps  # integer ndarray
        self._nnz = sum(b.size for b in blocks)
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
    def nnz(self):
        return self._nnz

    def copy(self):
        blocks = tuple(b.copy() for b in self._blocks)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def __add__(self, other):
        assert type(other) == type(self), "Mixing incompatible types"
        assert (
            self._axis_reps == other._axis_reps
        ), "SymmetricTensors have non-compatible axes"
        assert (
            self._n_row_legs == other._n_row_legs
        ), "SymmetricTensors have non-compatible fusion trees"
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
        return self._symmetry.combine_representations(
            *self._axis_reps[: self._n_leg_rows]
        )

    def get_column_representation(self):
        return self._symmetry.combine_representations(
            *self._axis_reps[: self._n_leg_rows]
        )

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

    def permutate(self, axes, n_leg_rows):  # signature != ndarray.transpose
        return NotImplemented

    def __matmul__(self, other):
        # do not construct empty blocks: those will be missing TODO: change this
        assert (
            self._axis_irreps[self._n_leg_rows :]
            == other._axis_irreps[: other._n_leg_rows]
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

        block_irreps = tuple(block_irreps)
        axis_reps = (
            self._axis_reps[: self._n_leg_rows]
            + other._axis_irreps[other._n_leg_rows :]
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

    _symmetry = AsymRepresentation
    _blocks_irreps = (np.zeros((1,), dtype=np.int8),)

    @classmethod
    def from_dense(cls, arr, n_leg_rows):
        axis_reps = tuple(AsymRepresentation(np.array([d])) for d in arr.shape)
        matrix_shape = (
            np.prod(arr.shape[:n_leg_rows]),
            np.prod(arr.shape[n_leg_rows:]),
        )
        block = arr.reshape(matrix_shape)
        return cls(axis_reps, n_leg_rows, (block,), cls._block_irreps)

    def toarray(self):
        return self._blocks[0].reshape(self._shape)

    def permutate(self, axes, n_leg_rows):
        arr = self._blocks[0].reshape(self._shape).transpose(axes)
        return AsymmetricTensor.from_dense(arr, n_leg_rows)

    @property
    def T(self):
        return AsymmetricTensor(
            self._axis_reps[self._n_leg_rows :] + self._axis_reps[: self._n_leg_rows],
            self._ndim - self._n_leg_rows,
            (self._blocks[0].T,),
            self._block_irreps,
        )


class AbelianSymmetricTensor(SymmetricTensor):
    @classmethod
    def from_dense(cls, arr, axis_reps, n_leg_rows):
        assert arr.shape == tuple(rep.dim for rep in axis_reps)
        mat_sh = (np.prod(arr.shape[:n_leg_rows]), np.prod(arr.shape[n_leg_rows:]))
        row_irreps = cls._symmetry.combine_representations(*axis_reps[:n_leg_rows])
        col_irreps = cls._symmetry.combine_representations(*axis_reps[n_leg_rows:])
        row_sort = row_irreps.argsort(kind="mergesort")
        sorted_row_irreps = row_irreps[row_sort]
        col_sort = col_irreps.argsort(kind="mergesort")
        sorted_col_irreps = col_irreps[col_sort]
        row_blocks = (
            [0]
            + list((sorted_row_irreps[:-1] != sorted_row_irreps[1:]).nonzero()[0] + 1)
            + [mat_sh[0]]
        )
        col_blocks = (
            [0]
            + list((sorted_col_irreps[:-1] != sorted_col_irreps[1:]).nonzero()[0] + 1)
            + [mat_sh[1]]
        )

        blocks = []
        block_irreps = []
        rbi, cbi, rbimax, cbimax = 0, 0, len(row_blocks) - 1, len(col_blocks) - 1
        m = arr.reshape(mat_sh)
        while rbi < rbimax and cbi < cbimax:
            if sorted_row_irreps[row_blocks[rbi]] == sorted_col_irreps[col_blocks[cbi]]:
                ri = row_sort[row_blocks[rbi] : row_blocks[rbi + 1]]
                ci = col_sort[col_blocks[cbi] : col_blocks[cbi + 1]]
                blocks.append(np.ascontiguousarray(m[ri[:, None], ci]))
                block_irreps.append(sorted_row_irreps[row_blocks[rbi]])
                rbi += 1
                cbi += 1
            elif (
                sorted_row_irreps[row_blocks[rbi]] < sorted_col_irreps[col_blocks[cbi]]
            ):
                rbi += 1
            else:
                cbi += 1
        return cls(axis_reps, n_leg_rows, blocks, block_irreps)

    def get_row_representation(self):
        return self._symmetry.combine_representations(
            *self._axis_reps[: self._n_leg_rows]
        )

    def get_column_representation(self):
        return self._symmetry.combine_representations(
            *self._axis_reps[: self._n_leg_rows]
        )

    def toarray(self):
        # cumbersome dealing with absent blocks
        row_irreps = self._symmetry.combine_raw_irreps(
            *(rep.irreps for rep in self._axis_reps[: self._n_leg_rows])
        )
        col_irreps = self._symmetry.combine_raw_irreps(
            *(rep.irreps for rep in self._axis_reps[self._n_leg_rows :])
        )
        allowed_irreps = sorted(set(row_irreps).intersect(col_irreps))
        ar = np.zeros(self.matrix_shape)
        i, j = 0, 0
        for irr in allowed_irreps:
            k = bisect.bisect_left(self._block_irreps, irr)
            if k < self._nblocks and self._block_irreps[k] == irr:
                b = self._blocks[k]
                assert b.shape[0] == (row_irreps == irr).sum()
                assert b.shape[1] == (col_irreps == irr).sum()
                ar[i : i + b.shape[0], j : j + b.shape[1]] = b
                i += b.shape[0]
                j += b.shape[1]
            else:
                i += (row_irreps == irr).sum()
                j += (col_irreps == irr).sum()

        row_perm = row_irreps.argsort(
            kind="stable"
        ).argsort()  # reverse from_dense permutation
        col_perm = col_irreps.argsort(
            kind="stable"
        ).argsort()  # reverse from_dense permutation
        ar = ar[row_perm[:, None], col_perm[:, None]]
        return ar.reshape(self._shape)


class U1_SymmetricTensor(AbelianSymmetricTensor):
    _symmetry = U1_Representation

    @property
    def T(self):
        # arrows on axes are swapped, meanning representations get conjugated
        # Colors need to stay sorted for __matmul__, so reverse all block-related lists.
        blocks = tuple(b.T for b in reversed(self._blocks))
        block_irreps = -self._block_irreps[::-1]
        axis_reps = tuple(
            self._axis_reps[i].conjugate()
            for i in range(-self._ndim + self._n_leg_rows, self._n_leg_rows)
        )
        return U1_SymmetricTensor(
            axis_reps, self._ndim - self._n_leg_rows, blocks, block_irreps
        )

    def permutate(self, axes, n_leg_rows):
        # cast to dense to reshape.
        ar = self.toarray().transpose(axes)
        axis_reps = tuple(self._axis_reps[i] for i in axes)
        return U1_SymmetricTensor.from_dense(ar, axis_reps, n_leg_rows)
