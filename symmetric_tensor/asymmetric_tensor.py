import numpy as np
import scipy.linalg as lg

from .symmetric_tensor import SymmetricTensor


class AsymmetricTensor(SymmetricTensor):
    """
    Tensor with no symmetry, mostly for debug and benchmarks purposes.

    An asymmetric representation is just a rank-0, size-1 integer array corresponding to
    its dimension.
    """

    # not a subclass of AbelianSymmetricTensor

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    @classmethod
    @property
    def symmetry(cls):
        return "trivial"

    _irrep = np.zeros([1], dtype=np.int8)

    @classmethod
    @property
    def singlet(cls):
        return np.ones((1,), dtype=int)

    @staticmethod
    def combine_representations(reps, signature):
        return np.prod([r for r in reps], axis=0)

    @staticmethod
    def conjugate_irrep(irr):
        return irr

    @staticmethod
    def conjugate_representation(rep):
        return rep

    @staticmethod
    def init_representation(degen, irreps):
        return degen.reshape((1,))

    @staticmethod
    def representation_dimension(rep):
        return rep[0]

    @staticmethod
    def irrep_dimension(rep):
        return 1

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        if signature is None:
            signature = np.zeros((len(row_reps) + len(col_reps)), dtype=bool)
            signature[len(row_reps) :] = True
        assert arr.shape == tuple(row_reps) + tuple(col_reps)
        block = arr.reshape(np.prod(row_reps), np.prod(col_reps))
        return cls(row_reps, col_reps, (block,), cls._irrep, signature)

    def toarray(self, as_matrix=False):
        if as_matrix:
            return self._blocks[0]
        return self._blocks[0].reshape(self._shape)

    @property
    def T(self):
        blocks = (self._blocks[0].T,)
        s = self._signature[np.arange(-self._ndim + self._nrr, self._nrr) % self._ndim]
        return type(self)(self._col_reps, self._row_reps, blocks, self._irrep, s)

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

        # generic case: goes back to tensor shape and transpose
        perm = np.array(row_axes + col_axes)
        arr = self._blocks[0].reshape(self._shape).transpose(perm)
        row_reps = tuple(np.array([d]) for d in arr.shape[: len(row_axes)])
        col_reps = tuple(np.array([d]) for d in arr.shape[len(row_axes) :])
        signature = self._signature[perm]
        return type(self).from_array(arr, row_reps, col_reps, signature)

    def group_conjugated(self):
        return type(self)(
            self._row_reps, self._col_reps, self._blocks, self._irrep, ~self._signature
        )

    def check_blocks_fit_representations(self):
        assert self._block_irreps == type(self)._irrep
        assert self._nblocks == 1
        assert len(self._blocks) == 1
        assert self._blocks[0].shape == self.matrix_shape
        assert self._shape == self._row_reps + self._col_reps
        return True

    def norm(self):
        return lg.norm(self._blocks[0])

    def totrivial(self):
        return self

    def update_signature(self, sign_update):
        # in the asymmetric case, bending an index to the left or to the right makes no
        # difference, signs can be ignored.
        up = np.asarray(sign_update, dtype=bool)
        assert up.shape == (self._ndim,)
        self._signature = self._signature ^ up
        assert self.check_blocks_fit_representations()
