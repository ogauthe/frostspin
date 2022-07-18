import numpy as np
import numba

from .abelian_symmetric_tensor import AbelianSymmetricTensor


@numba.njit
def _numba_combine_U1(reps, signature):
    combined = np.int8(1 - 2 * signature[0]) * reps[0]
    for r, s in zip(reps[1:], signature[1:]):
        rs = np.int8(1 - 2 * s) * r
        combined = (combined.reshape(-1, 1) + rs).ravel()
    return combined


class U1_SymmetricTensor(AbelianSymmetricTensor):
    """
    SymmetricTensor with global U(1) symmetry.
    """

    # most of the code is in AbelianSymmetricTensor, group fusion rules and conjugation
    # are the only symmetry-specific methods left.

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    @classmethod
    @property
    def symmetry(cls):
        return "U1"

    @staticmethod
    def combine_representations(reps, signature):
        assert signature.shape == (len(reps),)
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_U1(tuple(reps), signature)
        return (1 - 2 * signature[0]) * reps[0]

    @staticmethod
    def conjugate_representation(rep):
        return -rep

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    def toU1(self):
        return self
