import numpy as np
import scipy.sparse as ssp
import numba

from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor
from groups.su2_representation import SU2_Representation  # TODO remove me


def _get_projector(in1, in2, max_irrep=2 ** 30):
    # max_irrep cannot be set to None since irr3 loop depends on it
    degen, irreps = _numba_elementary_combine_SU2(in1[0], in1[1], in2[0], in2[1])
    trunc = irreps.searchsorted(max_irrep + 1)
    degen = degen[:trunc]
    irreps = irreps[:trunc]
    out_dim = degen @ irreps
    shift3 = np.zeros(irreps[-1] + 1, dtype=int)
    n = 0
    row = []
    col = []
    data = []
    for d3, irr3 in zip(degen, irreps):
        shift3[irr3] = n  # indexed with IRREP, not index
        n += d3 * irr3
    cs1 = [0, *(in1[0] * in1[1]).cumsum()]  # remember where to restart in in1
    cs2 = [0, *(in2[0] * in2[1]).cumsum()]  # remember where to restart in in2
    for i1, irr1 in enumerate(in1[1]):
        for i2, irr2 in enumerate(in2[1]):
            d2 = in2[0, i2]
            ar = np.arange(d2)
            sl2 = np.arange(cs2[i2], cs2[i2] + d2 * irr2)[:, None] * out_dim
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_irrep + 1), 2):
                p123 = SU2_Representation.elementary_projectors[irr1, irr2, irr3]
                sh = (irr1, d2, irr2, d2, irr3)
                temp = np.zeros(sh)
                temp[:, ar, :, ar] = p123
                temp = temp.reshape(irr1, d2 ** 2 * irr2 * irr3)
                row123, col123 = temp.nonzero()
                data123 = temp[row123, col123]
                shift1 = cs1[i1]
                for d1 in range(in1[0, i1]):
                    full_col = (
                        sl2 + np.arange(shift3[irr3], shift3[irr3] + d2 * irr3)
                    ).ravel()
                    row.extend(shift1 + row123)
                    col.extend(full_col[col123])
                    data.extend(data123)
                    shift3[irr3] += d2 * irr3
                    shift1 += irr1
    sh = (in1[0] @ in1[1], in2[0] @ in2[1] * out_dim)  # contract 1st leg in chained
    return ssp.csr_matrix((data, (row, col)), shape=sh)


def _get_projector_chained(*rep_in, singlet_only=False):
    r"""
    Tree structure: only first leg has depth
                product
                  /
                ...
                /
               /\
              /  \
             /\   \
            /  \   \
           1    2   3 ...
    """
    forwards, backwards = [[rep_in[0]], [rep_in[-1]]]
    n = len(rep_in)
    if n == 1:
        return ssp.eye(rep_in[0].dim).tocsc()

    for i in range(1, n):
        forwards.append(_numba_combine_SU2(forwards[i - 1], rep_in[i]))
        backwards.append(_numba_combine_SU2(backwards[i - 1], rep_in[-i - 1]))

    if singlet_only:
        # projection is made only on singlet. Remove irreps that wont fuse to 1.
        if forwards[-1][1, 0] != 1:
            raise ValueError("No singlet in product")
        truncations = [1]
        forwards[-1] = forwards[-1][:, : forwards[1].searchsorted(2)]
        for i in range(n - 1):
            trunc = backwards[i][1, -1]
            forwards[-i - 2] = forwards[-i][:, : forwards[1].searchsorted(trunc + 1)]
            truncations.append(trunc)
    else:
        truncations = [2 ** 30] * n

    proj = _get_projector(forwards[0], rep_in[1], max_irrep=truncations[-2])
    for (f, rep, trunc) in zip(forwards[1:], rep_in[2:], reversed(truncations[:-2])):
        p = _get_projector(f, rep, max_irrep=trunc)
        proj = proj.reshape(-1, p.shape[0]) @ p
    proj = proj.reshape(-1, forwards[-1][0] @ forwards[-1][1])
    return proj.tocsc()  # need to slice columns


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

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
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

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    _unitary_dic = {}

    @classmethod
    def irrep_dimension(cls, irr):
        return irr

    @classmethod
    def construct_matrix_projector(cls, row_reps, col_reps, conjugate_columns=False):
        repL = cls.combine_representations(*row_reps)
        repR = cls.combine_representations(*col_reps)
        dimLR = cls.representation_dimension(repL) * cls.representation_dimension(repR)
        projL = _get_projector_chained(*row_reps)
        projR = _get_projector_chained(*col_reps)

        target = sorted(set(repL[1]).intersection(repR[1]))
        if not target:
            raise ValueError("Representations have no common irrep")
        indL = repL[1].searchsorted(target)
        indR = repR[1].searchsorted(target)

        row = []
        col = []
        data = []
        shiftL = np.hstack((0, repL[0] * repL[1])).cumsum()
        shiftR = np.hstack((0, repR[0] * repR[1])).cumsum()
        shift_out = 0
        for i, irr in enumerate(target):
            degenL = repL[0, indL[i]]
            degenR = repR[0, indR[i]]
            matR = projR[:, shiftR[indR[i]] : shiftR[indR[i] + 1]]
            matR = (matR.reshape(-1, irr).T / np.sqrt(irr)).tocsr()
            if not conjugate_columns:
                sing_proj = ssp.csr_matrix(
                    SU2_Representation.irrep(irr).get_conjugator()
                )
                matR = sing_proj @ matR
            # it is not memory efficient to contract directly with the full matL: in
            # csr, indptr has size nrows, which would be dimL * degenL, much too large
            # (saturates memory). It also requires some sparse transpose. Using csc just
            # puts the problem on matR instead of matL. So to save memory, slice projL
            # irrep by irrep instead of taking all of them with degenL * irr. Slower but
            # memory efficient.
            for j in range(shiftL[indL[i]], shiftL[indL[i]] + degenL * irr, irr):
                matLR = projL[:, j : j + irr].tocsr()  # avoid large indptr
                matLR = matLR @ matR
                matLR = matLR.tocoo().reshape(dimLR, degenR)  # force coo cast
                row.extend(matLR.row)
                col.extend(shift_out + matLR.col)
                data.extend(matLR.data)
                shift_out += degenR

        assert shift_out == repL[0, indL] @ repR[0, indR]
        full_proj = ssp.csr_matrix((data, (row, col)), shape=(dimLR, shift_out))
        return full_proj

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    def group_conjugated(self):
        return self  # every SU(2) representations are self-conjugate
