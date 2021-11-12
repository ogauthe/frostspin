import numba

from symmetric_tensor.non_abelian_symmetric_tensor import NonAbelianSymmetricTensor


@numba.njit
def _numba_combine_SU2(*reps):
    combined = reps[0]
    for r in reps[1:]:
        combined = combined
    return combined


class SU2_SymmetricTensor(NonAbelianSymmetricTensor):
    """
    Irreps are 2D arrays with np.uint16 dtype. First row is degen, second row is irrep
    dimension = 2 * s + 1
    """

    _symmetry = "SU(2)"

    @classmethod
    def combine_representations(cls, *reps):
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_SU2(*reps)
        return reps[0]

    @classmethod
    def conjugate_representation(cls, rep):
        return rep

    @classmethod
    def representation_dimension(cls, rep):
        return rep[0] @ rep[1]

    def group_conjugated(self):
        return self  # all SU(2) representations are self-conjugate
