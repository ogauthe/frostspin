import numpy as np
import numba

from .abelian_symmetric_tensor import AbelianSymmetricTensor


@numba.njit(parallel=True)
def _numba_combine_U1(reps, signature):
    nx = len(reps)
    signs = (1 - 2 * signature).astype(np.int8)
    mod = np.array([r.size for r in reps], dtype=np.uint64)
    strides = np.ones((nx,), dtype=np.uint64)
    strides[1:] = mod[-1:0:-1]
    strides = strides.cumprod()[::-1].copy()
    combined = np.empty((strides[0] * mod[0],), dtype=np.int8)
    for i in numba.prange(combined.size):
        sz = 0
        for j in range(nx):
            ind = i // strides[j] % mod[j]
            sz += signs[j] * reps[j][ind]
        combined[i] = sz
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
        if len(reps) > 1:
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
