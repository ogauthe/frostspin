import numba

from symmetric_tensor.abelian_symmetric_tensor import AbelianSymmetricTensor


@numba.njit(cache=True)
def _numba_combine_U1(*reps):
    combined = reps[0]
    for r in reps[1:]:
        combined = (combined.reshape(-1, 1) + r).ravel()
    return combined


class U1_SymmetricTensor(AbelianSymmetricTensor):
    _symmetry = "U(1)"

    @classmethod
    def combine_representations(cls, *reps):
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_U1(*reps)
        return reps[0]

    @classmethod
    def conjugate_representation(cls, rep):
        return -rep
