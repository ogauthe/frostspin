import numpy as np
import scipy.linalg as lg

from symmetric_tensor.symmetric_tensor import SymmetricTensor


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
    def from_array(cls, arr, row_reps, col_reps, conjugate_columns=True):
        assert arr.shape == row_reps + col_reps
        block = arr.reshape(np.prod(row_reps), np.prod(col_reps))
        return cls(row_reps, col_reps, (block,), cls._irrep)

    def toarray(self, as_matrix=False):
        if as_matrix:
            return self._blocks[0]
        return self._blocks[0].reshape(self._shape)

    def permutate(self, row_axes, col_axes):
        arr = self._blocks[0].reshape(self._shape).transpose(row_axes + col_axes)
        n = len(row_axes)
        return type(self).from_array(arr, arr.shape[:n], arr.shape[n:])

    def group_conjugated(self):
        return self

    def norm(self):
        return lg.norm(self._blocks[0])
