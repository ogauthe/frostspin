import numpy as np
import scipy.linalg as lg


class Abelian_Representation(object):
    """
    Minimalist class for Abelian representations. A non-abelian symmetry is a much more
    complicate object with quite different features. Implementation becomes simpler
    without defining a common Representation base class.
    """

    def __init__(self, degen, irreps):
        """
        Construct an abelian representation.

        Parameters
        ----------
        degen : integer array
            Degeneracy of given irreps
        irreps : tuple of Irrep
            Irreps needs to implement __eq__. No other requirement to keep things
            simple.
        """
        self._degen = degen
        self._irreps = irreps
        self._dim = degen.sum()
        self._n_irreps = degen.size
        assert len(irreps) == self._n_irreps
        assert degen.shape == (self._n_irreps,)
        assert degen.all()
        assert np.issubdtype(degen.dtype, np.integer)

    @property
    def dim(self):
        return self._dim

    @property
    def degen(self):
        return self._degen

    @property
    def irreps(self):
        return self._irreps

    def __eq__(self, other):
        return self._irreps == other._irreps and (self._degen == other._degen).all()


class SymmetricTensor(object):
    """
    Generic base class to deal with symmetric tensors. Defines interface that can be
    implemented as derived classes for any symmetry.

    Tensors are seen as matrices, with legs grouped into two packs, rows and columns.
    Some authorized blocks are allowed to be missing if they are 0. Such cases arises
    from a matmul call when irrep X appears on left rows and right columns but not in
    the middle representation.
    """

    def __init__(self, axis_reps, n_leg_rows, blocks, block_irreps):
        self._axis_reps = axis_reps
        self._n_leg_rows = n_leg_rows
        self._shape = tuple(rep.dim for rep in axis_reps)
        self._ndim = len(axis_reps)
        self._nblocks = len(blocks)
        self._blocks = blocks
        self._block_irreps = block_irreps
        self._nnz = sum(b.size for b in blocks)
        assert 0 < n_leg_rows < self._ndim
        assert len(block_irreps) == self._nblocks

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
    def nnz(self):
        return self._nnz

    def copy(self):
        blocks = tuple(b.copy() for b in self._blocks)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    @property
    def T(self):
        # Transpose the matrix representation of the tensor, ie swap rows and columns
        # and transpose diagonal blocks, without any data move or copy. Irreps need to
        # be conjugate since row (bra) and columns (ket) are swapped. If this operation
        # is non-trivial, specialization is required.

        # other solution: have irreps object supporting < and conjugate
        # then one can have
        # block_irreps = self._block_irreps.conj()
        # so = block_irreps.argsort()
        # block_irreps = block_irreps[so]
        # blocks = tuple(self._blocks[i].T for i in so)
        # perm = tuple(range(ndim - n_leg_rows, ndim)) + tuple(range(ndim - n_leg_rows))
        # axis_rep = self._axis_reps.conj()[perm]
        # and no need for specialization  => too heavy for python
        blocks = tuple(b.T for b in self._blocks)
        axis_reps = (
            self._axis_reps[self._n_leg_rows :] + self._axis_reps[: self._n_leg_rows]
        )
        return type(self)(
            axis_reps, self._ndim - self._n_leg_rows, blocks, self._block_irreps
        )

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
        return NotImplemented

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

    def __neg__(self):
        blocks = tuple(-b for b in self._blocks)
        return type(self)(self._axis_reps, self._n_leg_rows, blocks, self._block_irreps)

    def toarray(self):
        return NotImplemented  # symmetry dependant

    def transpose(self, axes, n_leg_rows):
        return NotImplemented  # symmetry dependant

    def __matmul__(self, other):
        # requires __lt__ to be defined for irreps
        # do not construct empty blocks: those will be missing
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

        block_irreps = tuple(self._block_irreps)
        axis_reps = (
            self._axis_reps[: self._n_leg_rows]
            + other._axis_irreps[other._n_leg_rows :]
        )
        return type(self)(axis_reps, self._n_leg_rows, blocks, block_irreps)

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
            # TODO implement sparse svd
            try:
                block_u[bi], block_s[bi], block_v[bi] = lg.svd(
                    b, full_matrices=False, check_finite=False
                )
            except lg.LinAlgError as err:
                print("Error in scipy dense SVD:", err)
                block_u[bi], block_s[bi], block_v[bi] = lg.svd(
                    b, full_matrices=False, check_finite=False, driver="gesvd"
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
        mid_rep = type(self._axis_reps)(block_cuts, mid_irreps)
        rep_u = self._axis_reps[: self._n_leg_rows] + (mid_rep,)
        rep_v = (mid_rep,) + self._axis_reps[self._n_leg_rows :]

        U = type(self)(rep_u, self._n_leg_rows, block_u, mid_rep.irreps)
        V = type(self)(rep_v, 1, block_v, mid_rep.irreps)
        s = np.array(s[::-1])
        return U, s, V


# class AsymmetricTensor(SymmetricTensor):
