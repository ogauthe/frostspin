import numpy as np
import numba

from .abelian_symmetric_tensor import AbelianSymmetricTensor


@numba.njit
def _numba_combine_U1(reps, signature):
    nx = len(reps)
    c0 = np.int8(1 - 2 * signature[0]) * reps[0]
    for i in range(1, nx):
        n = reps[i].size
        rs = (1 - 2 * signature[i]) * reps[i]
        c1 = np.empty((c0.size * n,), dtype=np.int8)
        for j in range(c0.size):
            for k in range(n):
                c1[j * n + k] = c0[j] + rs[k]
        c0 = c1
    return c0


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
