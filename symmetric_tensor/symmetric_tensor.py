import numpy as np
import scipy.linalg as lg


class SymmetricTensor(object):
    """
    Generic base class to deal with symmetric tensors. Defines interface that can be
    implemented as derived classes for any symmetry.

    Tensors are seen as matrices, with legs grouped into two packs, rows and columns.
    """

    def __init__(self, blocks, block_irreps, shape, axes_irreps, n_leg_rows):
        self._ndim = len(shape)
        self._nblocks = len(blocks)
        assert len(block_irreps) == self._nblocks
        # for non abelian symmetries, len(axis_irreps) != dim(axis)
        assert len(axes_irreps) == self._ndim
        assert 0 < n_leg_rows < self._ndim
        self._blocks = blocks
        self._block_irreps = block_irreps
        self._shape = shape
        self._axes_irreps = axes_irreps
        self._n_leg_rows = n_leg_rows
        self._ncoeff = sum(b.size for b in blocks)

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
    def n_leg_rows(self):
        return self._n_leg_rows

    @property
    def ncoeff(self):
        return self._ncoeff

    def toarray(self):
        return NotImplemented

    @property
    def T(self):
        # pretty different from transpose, so write own implementation
        blocks = [b.T for b in self._blocks]
        shape = self._shape[self._n_leg_rows :] + self._shape[: self._n_leg_rows]
        axes_irreps = (
            self._axes_irreps[self._n_leg_rows :]
            + self._axes_irreps[: self._n_leg_rows]
        )
        return type(self)(
            blocks,
            self._block_irreps,
            shape,
            axes_irreps,
            self._ndim - self._n_leg_rows,
        )

    def transpose(self, axes, n_leg_rows):
        return NotImplemented
        """
        if len(axes) != self._ndim:
            raise ValueError("axes don't match SymmetricTensor")
        if tuple(axes) == tuple(range(self._ndim)) and n_leg_rows == self._n_leg_rows:
            return self
        """

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

    def __neg__(self):
        blocks = [-b for b in self._blocks]
        return type(self)(
            blocks,
            self._block_irreps,
            self._shape,
            self._axes_irreps,
            self._n_leg_rows,
        )

    def __matmul__(self, other):
        # unfortunately dealing with empty blocks requires knowledge on symmetry
        return NotImplemented
        if (
            self._axes_irreps[self._n_leg_rows :]
            != other._axes_irreps[: other._n_leg_rows]
        ):
            raise ValueError("SymmetricTensors have non-compatible axes")
        for (b1, b2) in zip(self._blocks, other._blocks):
            blocks = tuple(b1 @ b2 for (b1, b2) in zip(self._blocks, other._blocks))
        block_irreps = tuple(self._block_irreps)
        shape = self._shape[: self._n_leg_rows] + other._shape[other._n_leg_rows :]
        axes_irreps = (
            self._axes_irreps[: self._n_leg_rows]
            != other._axes_irreps[other._n_leg_rows :]
        )
        return type(self)(
            blocks,
            block_irreps,
            shape,
            axes_irreps,
            self._n_leg_rows,
        )

    def __mul__(self, x):
        return NotImplemented

    def __add__(self, other):
        if self._axes_irreps != other._axes_irreps:
            raise ValueError("SymmetricTensors have non-compatible axes")
        if self._n_row_legs != other._n_row_legs:
            raise ValueError("SymmetricTensors have non-compatible matrix shapes")
        blocks = tuple(b1 + b2 for (b1, b2) in zip(self._blocks, other._blocks))
        return type(self)(
            blocks,
            self._block_irreps,
            self._shape,
            self._axes_irreps,
            self._n_leg_rows,
        )

    def svd(self, cut=None, rcutoff=0.0):
        """
        Compute block-wise SVD of self and keep only cut largest singular values. Do not
        truncate if cut is not provided. Keep only values larger than rcutoff * max(sv).
        """
        block_u = [None] * self._nblocks
        block_s = [None] * self._nblocks
        block_v = [None] * self._nblocks
        block_max_vals = np.empty(self._nblocks)
        for bi, b in enumerate(self._blocks):
            block_u[bi], block_s[bi], block_v[bi] = lg.svd(
                b, full_matrices=False, check_finite=False
            )
            block_max_vals[bi] = block_s[bi][0]

        cutoff = block_max_vals.max() * rcutoff  # cannot be set before 1st loop
        block_cuts = [0] * self._nblocks
        if cut is None:
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
        for bi in reversed(range(self._nblocks)):  # reversed to del
            if block_cuts[bi]:
                block_u[bi] = block_u[bi][:, : block_cuts[bi]]
                s.extend(block_s[bi][: block_cuts[bi]][::-1])
                block_v[bi] = block_v[bi][: block_cuts[bi]]
            else:  # do not keep empty matrices
                del block_u[bi]
                del block_v[bi]

        mid_rep = (block_cuts, self._block_irreps)  # TODO
        U = type(self)(block_u, mid_rep.irreps, self._left_rep, mid_rep)
        V = type(self)(block_v, mid_rep.irreps, mid_rep, self._right_rep)
        s = np.array(s[::-1])
        return U, s, V, mid_rep

    def copy(self):
        blocks = tuple(b.copy() for b in self._blocks)
        return type(self)(
            blocks,
            self._block_irreps,
            self._shape,
            self._axes_irreps,
            self._n_leg_rows,
        )


# class AsymmetricTensor(SymmetricTensor):
