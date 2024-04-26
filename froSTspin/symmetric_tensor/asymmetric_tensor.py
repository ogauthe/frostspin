import numpy as np

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
    _symmetry = "trivial"
    _irrep = np.zeros([1], dtype=np.int8)

    @staticmethod
    def singlet():
        return np.ones((1,), dtype=int)

    @staticmethod
    def combine_representations(reps, signature):
        return np.prod(reps, axis=0)

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
        return int(rep)

    @staticmethod
    def irrep_dimension(rep):
        return 1

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################

    @classmethod
    def get_block_sizes(cls, row_reps, col_reps, signature):
        nr = np.prod(row_reps)
        nc = np.prod(col_reps)
        return cls._irrep, np.array([[nr, nc]])

    @classmethod
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        if signature is None:
            signature = np.zeros((len(row_reps) + len(col_reps)), dtype=bool)
            signature[len(row_reps) :] = True
        assert arr.shape == tuple(row_reps) + tuple(col_reps)
        block = arr.reshape(np.prod(row_reps), np.prod(col_reps))
        return cls(row_reps, col_reps, (block,), cls._irrep, signature)

    def _tomatrix(self):
        return self._blocks[0]

    def _transpose_data(self):
        return (self._blocks[0].T,), self._irrep

    def _permute_data(self, axes, nrr):
        arr = self._blocks[0].reshape(self._shape).transpose(axes)
        sh = (np.prod(arr.shape[:nrr]), np.prod(arr.shape[nrr:]))
        arr = arr.reshape(sh)
        return (arr,), self._irrep

    def dual(self):
        return type(self)(
            self._row_reps, self._col_reps, self._blocks, self._irrep, ~self._signature
        )

    def check_blocks_fit_representations(self):
        assert self._block_irreps.shape == (1,)
        assert self._block_irreps[0] == 0
        assert self._nblocks == 1
        assert len(self._blocks) == 1
        assert self._blocks[0].shape == self.matrix_shape
        assert self._shape == self._row_reps + self._col_reps
        return True

    def totrivial(self):
        return self

    def update_signature(self, sign_update):
        # in the asymmetric case, bending an index to the left or to the right makes no
        # difference, signs can be ignored.
        up = np.asarray(sign_update, dtype=bool)
        assert up.shape == (self._ndim,)
        self._signature = self._signature ^ up
        assert self.check_blocks_fit_representations()
