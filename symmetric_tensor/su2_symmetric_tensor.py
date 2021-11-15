import numpy as np
import numba

from symmetric_tensor.non_abelian_symmetric_tensor import NonAbelianSymmetricTensor
from groups.su2_representation import SU2_Representation  # TODO remove me
import groups.su2_matrix  # TODO remove me


@numba.njit
def _numba_elementary_combine_SU2(degen1, irreps1, degen2, irreps2):
    degen = np.zeros(irreps1[-1] + irreps2[-1] - 1, dtype=np.int64)
    for (d1, irr1) in zip(degen1, irreps1):
        for (d2, irr2) in zip(degen2, irreps2):
            for irr in range(abs(irr1 - irr2), irr1 + irr2 - 1, 2):
                degen[irr] += d1 * d2  # shit irr-1 <-- irr to start at 0
    nnz = degen.nonzero()[0]
    return degen[nnz], nnz + 1


@numba.njit
def _numba_combine_SU2(*reps):
    degen, irreps = reps[0]
    for r in reps[1:]:
        degen, irreps = _numba_elementary_combine_SU2(degen, irreps, r[0], r[1])
    return np.concatenate((degen, irreps)).reshape(2, -1)


class SU2_SymmetricTensor(NonAbelianSymmetricTensor):
    """
    Irreps are 2D arrays with int dtype. First row is degen, second row is irrep
    dimension = 2 * s + 1
    """

    _symmetry = "SU(2)"
    _unitary_dic = {}

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

    @classmethod
    def irrep_dimension(cls, irr):
        return irr

    def group_conjugated(self):
        return self  # all SU(2) representations are self-conjugate

    @classmethod
    def construct_matrix_projector(cls, row_reps, col_reps, conjugate_columns=False):
        # WIP
        # reuse SU2_Matrix stuff
        rep_left_enum = tuple(SU2_Representation(r[0], r[1]) for r in row_reps)
        rep_right_enum = tuple(SU2_Representation(r[0], r[1]) for r in col_reps)
        return groups.su2_matrix.construct_matrix_projector(
            rep_left_enum, rep_right_enum, conj_right=conjugate_columns
        )
