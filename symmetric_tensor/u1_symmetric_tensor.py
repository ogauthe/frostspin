import numba

from .abelian_symmetric_tensor import AbelianSymmetricTensor


@numba.njit
def _numba_combine_U1(*reps):
    combined = reps[0]
    for r in reps[1:]:
        combined = (combined.reshape(-1, 1) + r).ravel()
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
    _symmetry = "U(1)"

    @staticmethod
    def combine_representations(*reps):
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_U1(*reps)
        return reps[0]

    @staticmethod
    def conjugate_representation(rep):
        return -rep
