import numpy as np
import scipy.linalg as lg

from symmetric_tensor.symmetric_tensor import SymmetricTensor


class AsymmetricTensor(SymmetricTensor):
    """
    Tensor with no symmetry, mostly for debug and benchmarks purposes.

    An asymmetric representation is just a rank-0, size-1 integer array corresponding to
    its dimension.
    """

    # not a subclass of AbelianSymmetricTensor
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
    def from_array(cls, arr, row_reps, col_reps, conjugate_columns=True):
        assert arr.shape == tuple(row_reps) + tuple(col_reps)
        block = arr.reshape(np.prod(row_reps), np.prod(col_reps))
        return cls(row_reps, col_reps, (block,), cls._irrep)

    def _toarray(self):
        return self._blocks[0]

    def _permutate(self, row_axes, col_axes):
        arr = self._blocks[0].reshape(self._shape).transpose(row_axes + col_axes)
        row_reps = tuple(np.array(d) for d in arr.shape[: len(row_axes)])
        col_reps = tuple(np.array(d) for d in arr.shape[len(row_axes) :])
        return type(self).from_array(arr, row_reps, col_reps)

    def group_conjugated(self):
        return self

    def norm(self):
        return lg.norm(self._blocks[0])
