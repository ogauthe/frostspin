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

    def conjugate(self):
        blocks = (self._blocks[0].conj(),)
        return AsymmetricTensor(self._axis_reps, self._n_leg_rows, blocks, self._irrep)

    def norm(self):
        return lg.norm(self._blocks[0])
