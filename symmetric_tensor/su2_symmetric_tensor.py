import numpy as np
import scipy.sparse as ssp
import scipy.linalg as lg
import numba

from .lie_group_symmetric_tensor import LieGroupSymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor
from .o2_symmetric_tensor import O2_SymmetricTensor
from groups.su2_representation import SU2_Representation  # TODO remove me

_singlet = np.array([[1], [1]])


def _get_projector(in1, in2, s1, s2, max_irrep=2**30):
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
        diag1 = (np.arange(irr1 % 2, irr1 + irr1 % 2) % 2 * 2 - 1)[:, None, None]
        for i2, irr2 in enumerate(in2[1]):
            diag2 = (np.arange(irr2 % 2, irr2 + irr2 % 2) % 2 * 2 - 1)[None, :, None]
            d2 = in2[0, i2]
            ar = np.arange(d2)
            sl2 = np.arange(cs2[i2], cs2[i2] + d2 * irr2)[:, None] * out_dim
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_irrep + 1), 2):
                p123 = SU2_Representation.elementary_projectors[irr1, irr2, irr3]
                # apply spin-reversal operator according to signatures
                if s1:
                    p123 = p123[::-1] * diag1
                if s2:
                    p123 = p123[:, ::-1] * diag2
                sh = (irr1, d2, irr2, d2, irr3)
                temp = np.zeros(sh)
                temp[:, ar, :, ar] = p123
                temp = temp.reshape(irr1, d2**2 * irr2 * irr3)
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


def _get_projector_chained(rep_in, signature, singlet_only=False):
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
    n = len(rep_in)
    if n == 1:
        r = rep_in[0]
        if singlet_only:
            if r[1, 0] != 1:
                raise ValueError("No singlet in product")
            return ssp.eye(r[0] @ r[1], r[0, 0])
        return _get_projector(r, _singlet, signature[0], False)  # care for conj

    forwards = [rep_in[0]]
    for i in range(1, n):
        forwards.append(_numba_combine_SU2(forwards[i - 1], rep_in[i]))

    if singlet_only:
        if forwards[-1][1, 0] != 1:
            raise ValueError("No singlet in product")
        # projection is made only on singlet. Remove irreps that wont fuse to 1.
        backwards = [rep_in[-1]]
        for i in range(1, n):
            backwards.append(_numba_combine_SU2(backwards[i - 1], rep_in[-i - 1]))
        truncations = [1]
        forwards[-1] = forwards[-1][:, : forwards[-1][1].searchsorted(2)]
        for i in range(n - 1):
            trunc = backwards[i][1, -1]
            cut = forwards[-i - 2][1].searchsorted(trunc + 1)
            forwards[-i - 2] = forwards[-i - 2][:, :cut]
            truncations.append(trunc)
    else:
        truncations = [2**30] * n

    proj = _get_projector(
        forwards[0], rep_in[1], signature[0], signature[1], max_irrep=truncations[-2]
    )
    for i in range(1, n - 1):
        p = _get_projector(
            forwards[i],
            rep_in[i + 1],
            False,
            signature[i + 1],
            max_irrep=truncations[-i - 2],
        )
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


class SU2_SymmetricTensor(LieGroupSymmetricTensor):
    """
    Irreps are 2D arrays with int dtype. First row is degen, second row is irrep
    dimension = 2 * s + 1
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    @classmethod
    @property
    def symmetry(cls):
        return "SU2"

    @classmethod
    @property
    def singlet(cls):
        return _singlet

    @staticmethod
    def combine_representations(reps, signature):
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_SU2(*reps)
        return reps[0]

    @staticmethod
    def conjugate_representation(rep):
        return rep

    @staticmethod
    def representation_dimension(rep):
        return rep[0] @ rep[1]

    @classmethod
    def irrep_dimension(cls, irr):
        return irr

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    _unitary_dic = {}

    @classmethod
    def construct_matrix_projector(cls, row_reps, col_reps, signature):
        nrr = len(row_reps)
        assert signature.shape == (nrr + len(col_reps),)
        repL = cls.combine_representations(row_reps, signature[:nrr])
        repR = cls.combine_representations(col_reps, signature[nrr:])
        dimLR = cls.representation_dimension(repL) * cls.representation_dimension(repR)
        projL = _get_projector_chained(row_reps, signature[:nrr])
        projR = _get_projector_chained(col_reps, ~signature[nrr:])

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
            matR = matR.reshape(-1, irr).T.tocsr() / np.sqrt(irr)

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
        signature = ~self._signature
        p = self.construct_matrix_projector(
            self._row_reps, self._col_reps, self._signature
        )
        p2 = self.construct_matrix_projector(self._row_reps, self._col_reps, signature)
        # can avoid computing p2 by taking kron( (-1)^(irr+1)) @ p
        # can also be stored in unitary dic with permutation = Id
        raw_data = p2.T @ (p @ self.to_raw_data())
        blocks, block_irreps = self._blocks_from_raw_data(
            raw_data, self.get_row_representation(), self.get_column_representation()
        )
        return type(self)(
            self._row_reps, self._col_reps, blocks, block_irreps, signature
        )

    def update_signature(self, sign_update):
        # In the non-abelian case, updating signature can be done from the left or from
        # right, yielding different results.
        up = np.asarray(sign_update, dtype=np.int8)
        assert up.shape == (self._ndim,)
        assert (np.abs(up) < 2).all()

        # only construct projector if needed, ie there is a -1
        # For this case, add diagonal -1 for half integer spins in legs with a loop
        if (up < 0).any():
            p = self.construct_matrix_projector(
                self._row_reps, self._col_reps, self._signature
            )
            diag = ssp.eye(1, format="coo")  # kron operates in coo format
            for i in range(self._ndim):
                d = self._shape[i]
                if up[i] < 0:
                    inds = np.arange(d)
                    coeff = np.ones((d,), dtype=int)
                    if i < self._nrr:
                        r = self._row_reps[i]
                    else:
                        r = self._col_reps[i - self._nrr]
                    k = 0
                    for degen, irr in r.T:
                        if not irr % 2:
                            coeff[k : k + degen * irr] = -1
                        k += degen * irr
                    m = ssp.coo_matrix((coeff, (inds, inds)), shape=(d, d))
                else:
                    m = ssp.eye(d, format="coo")
                diag = ssp.kron(diag, m)

            # could be stored in unitary_dic if too expensive, expect uncommon operation
            diag = diag.todia()  # we know the matrix is diagonal
            raw_data = p.T @ (diag @ (p @ self.to_raw_data()))
            blocks, block_irreps = self._blocks_from_raw_data(
                raw_data,
                self.get_row_representation(),
                self.get_column_representation(),
            )
            self._blocks = blocks
            self._block_irreps = block_irreps

        self._signature = self._signature ^ up.astype(bool)

    def toabelian(self):
        return self.toU1()

    def toU1(self):
        # efficient cast to U(1): project directly raw data to U(1) blocks
        # 1) construct U(1) representations
        reps = []
        for r in self._row_reps + self._col_reps:
            sz = np.empty((r[0] @ r[1],), dtype=np.int8)
            k = 0
            for (d, irr) in zip(r[0], r[1]):
                sz_irr = np.arange(irr - 1, -irr - 1, -2, dtype=np.int8)
                sz[k : k + d * irr].reshape(d, irr)[:] = sz_irr
                k += d * irr
            reps.append(sz)

        # 2) combine into row and column
        row_irreps = U1_SymmetricTensor.combine_representations(
            reps[: self._nrr], self._signature[: self._nrr]
        )
        col_irreps = U1_SymmetricTensor.combine_representations(
            reps[self._nrr :], ~self._signature[self._nrr :]
        )
        ncols = col_irreps.size

        # 3) find non-empty U(1) blocks directly from current block_irreps
        sze = (self._block_irreps % 2).nonzero()[0]  # integer spins
        szo = ((self._block_irreps + 1) % 2).nonzero()[0]  # half interger spins
        block_irreps = []
        if sze.size:
            sze_maxp1 = self._block_irreps[sze[-1]]
            block_irreps.extend(np.arange(-sze_maxp1 + 1, sze_maxp1, 2))
        if szo.size:
            szo_maxp1 = self._block_irreps[szo[-1]]
            block_irreps.extend(np.arange(-szo_maxp1 + 1, szo_maxp1, 2))
        block_irreps = np.sort(block_irreps)

        # 4) construct U(1) block by slicing Clebsh-Gordon projector
        blocks = []
        proj = self.construct_matrix_projector(
            self._row_reps, self._col_reps, self._signature
        )
        raw = self.to_raw_data()
        for sz in block_irreps:
            ri = (row_irreps == sz).nonzero()[0]
            ci = (col_irreps == sz).nonzero()[0]
            inds = ncols * ri[:, None] + ci
            b = (proj[inds.ravel()] @ raw).reshape(inds.shape)
            blocks.append(b)

        assert (
            abs(np.sqrt(sum(lg.norm(b) ** 2 for b in blocks)) - self.norm())
            < 1e-14 * self.norm()
        )
        return U1_SymmetricTensor(
            reps[: self._nrr], reps[self._nrr :], blocks, block_irreps, self._signature
        )

    def toO2(self):
        return O2_SymmetricTensor.from_SU2(self)
