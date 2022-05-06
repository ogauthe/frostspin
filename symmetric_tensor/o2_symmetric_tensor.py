import numpy as np
import scipy.sparse as sp
import numba

from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor


@numba.njit
def _numba_combine_O2(*reps):
    # 0o x n = n
    # 0e x n = n
    # n x n = 2n + 0e + 0o
    # n x m = (n+m) + |n-m|
    combined = None  # TODO
    return combined


def _O2_rep_to_U1(r):
    assert (r[1] > -2).all()
    ru1 = np.empty((O2_SymmetricTensor.representation_dimension(r),), dtype=np.int8)
    i = 0
    k = 0
    if ru1[1, 0] == -1:
        ru1[: r[0, 0]] = 0
        k += r[0, 0]
        i += 1
    if ru1[1, i] == 0:
        ru1[k : r[0, i]] = 0
        k += r[0, i]
        i += 1
    for j in range(i, r.shape[0]):
        ru1[k : k + r[0, i]] = r[1, i]
        k += r[0, i]
        ru1[k : k + r[0, i]] = -r[1, i]
        k += r[0, i]
    return ru1


@numba.njit
def _get_coo_proj(signs, so):
    ecoeff = []
    erows = []
    ecols = []
    ocoeff = []
    orows = []
    ocols = []
    ie, io = 0, 0
    state_indices = set(range(so.size))
    while state_indices:
        i = state_indices.pop()
        j = so[i]
        isign = signs[i]
        if i == j:  # fixed point
            if isign == 1:  # even
                ecoeff.append(1.0)
                erows.append(ie)
                ecols.append(i)
                ie += 1
            else:  # odd
                ocoeff.append(1.0)
                orows.append(io)
                ocols.append(i)
                io += 1
        else:
            state_indices.remove(j)
            ecoeff.append(1.0 / np.sqrt(2))
            erows.append(ie)
            ecols.append(i)
            ecoeff.append(isign / np.sqrt(2))
            erows.append(ie)
            ecols.append(j)
            ocoeff.append(1.0 / np.sqrt(2))
            orows.append(io)
            ocols.append(i)
            ocoeff.append(-isign / np.sqrt(2))
            orows.append(io)
            ocols.append(j)
            ie += 1
            io += 1

    return ie, ecoeff, erows, ecols, io, ocoeff, orows, ocols


def _get_b0_projectors(row_reps, col_reps):
    """
    Construct 0odd and 0even projectors. They allow to decompose U(1) Sz=0 sector into
    0odd (aka -1) and 0even (aka 0) sectors.
    """

    shr = np.array([O2_SymmetricTensor.representation_dimension(r) for r in row_reps])
    row_cp = np.array([1, *shr[1:]]).cumprod()[::-1]

    rsz_mat = (_O2_rep_to_U1(row_reps) == 0).nonzero()[0]  # find Sz=0 states
    rsz_t = (rsz_mat // row_cp[:, None]).T % shr  # multi-index form
    # rszb_t = rmap[rsz_t]  # map all indices to their spin reversed  # TODO
    rszb_t = None * rsz_t
    rszb_mat = rszb_t @ row_cp  # back to matrix form
    rso = rszb_mat.argsort()  # imposed sorted block indices in U(1) => argsort
    # rsign = raxis_signs[rsz_t].prod(axis=1)  # row signs  # TODO
    rsign = None
    ie, ecoeff, erows, ecols, io, ocoeff, orows, ocols = _get_coo_proj(rsign, rso)
    pre = sp.coo_matrix((ecoeff, (erows, ecols)), shape=(ie, rso.size))
    pro = sp.coo_matrix((ocoeff, (orows, ocols)), shape=(io, rso.size))
    pre = pre.tocsr()
    pro = pro.tocsr()

    # same operation for columns
    shc = [O2_SymmetricTensor.representation_dimension(r) for r in col_reps]
    col_cp = np.array([1, *shc[1:]]).cumprod()[::-1]
    csz_mat = (_O2_rep_to_U1(col_reps) == 0).nonzero()[0]
    csz_t = (csz_mat // col_cp[:, None]).T % shc
    # cszb_t = cmap[csz_t]  # TODO
    cszb_t = None * csz_t
    cszb_mat = cszb_t @ col_cp
    cso = cszb_mat.argsort()
    # csign = caxis_signs[csz_t].prod(axis=1)  # TODO
    csign = None
    ie, ecoeff, erows, ecols, io, ocoeff, orows, ocols = _get_coo_proj(csign, cso)
    pce = sp.coo_matrix((ecoeff, (erows, ecols)), shape=(ie, cso.size))
    pco = sp.coo_matrix((ocoeff, (orows, ocols)), shape=(io, cso.size))
    pce = pce.T.tocsr()
    pco = pco.T.tocsr()

    return pro, pre, pco, pce


class O2_SymmetricTensor(NonAbelianSymmetricTensor):
    """
    SymmetricTensor with global O(2) symmetry. Implement it as semi direct product of
    Z_2 and U(1). Irreps of U(1) are labelled by integer n. For n>0 even, one gets a
    dimension 2 irrep of O(2) by coupling n and -n. For n odd, this is a projective
    representation. The sector n=0 is split into 2 1D irreps, 0 even and 0 odd.

    irrep -1 labels 0 odd
    irrep 0 labels 0 even
    irrep n>0 labels irrep (+n, -n)
    """

    # impose consecutive +n and -n sectors
    # impose, for n even, same sign in +n and -n (differs from SU(2))
    # impose, for n odd, sign from +n to -n (differs from SU(2))

    # there are 2 possibilites to store reprensetation info
    # => mimic abelian, with format
    # [1, -1, 0, 1]
    # [o,  e, o, e]
    # but then where to store mapping on Sz-reversed?
    # could be a third row of rep, but then rep has to have int64 dtype => heavy

    # OR full non-abelian: rep = array([degen, irreps])
    # with irreps being -1 for 0odd, 0 for 0even, n for +/-n, which need to be taken
    # consecutive AND impose some rules like n even => even, n odd => sign appears from
    # Sz<0 to Sz>0
    # may have difficulties when transposing / taking adjoint
    # last possibility: third layer with sign
    # still impose consecutive n / -n, but allows for more flexibility in signs

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    @classmethod
    @property
    def symmetry(cls):
        return "O2"

    @staticmethod
    def representation_dimension(rep):
        return rep[0] @ ((rep[1] > 0).astype(int) + 1)

    @staticmethod
    def irrep_dimension(irrep):
        return irrep > 0 + 1

    @staticmethod
    def combine_representations(*reps):
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_O2(*reps)
        return reps[0]

    @staticmethod
    def conjugate_representation(rep):
        return rep

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, conjugate_columns=True):
        # not fully efficient: 1st construct U(1) symmetric tensor to get abelian blocks
        # then split sector 0 into 0even and 0odd
        # discard sectors with n < 0
        # keep sectors with n > 0 as they are
        u1_row_reps = []
        for r in row_reps:
            u1_row_reps.append(_O2_rep_to_U1(r))
        u1_col_reps = []
        for r in col_reps:
            u1_col_reps.append(_O2_rep_to_U1(r))

        tu1 = U1_SymmetricTensor.from_array(arr, u1_row_reps, u1_col_reps)
        return cls.from_U1(tu1, row_reps, col_reps)

    @classmethod
    def from_U1(cls, tu1, row_reps, col_reps):
        blocks = []
        block_irreps = []
        i0 = tu1.block_irreps.searchsorted(0)
        if tu1.block_irreps[i0] == 0:
            block_irreps.append(-1)
            block_irreps.append(0)
            pro, pre, pco, pce = _get_b0_projectors(row_reps, col_reps)
            blocks.append(pro @ tu1.blocks[i0] @ pco)
            blocks.append(pre @ tu1.blocks[i0] @ pce)
        block_irreps.extend(tu1.block_irreps[i0:])
        blocks.extend(tu1.block[i0:])
        return cls(row_reps, col_reps, blocks, block_irreps)

    def _toarray(self):
        return self.toU1().toarray()

    def toabelian(self):
        return self.toU1()

    def toU1(self):
        u1_row_reps = []
        for r in self._row_reps:
            u1_row_reps.append(_O2_rep_to_U1(r))
        u1_col_reps = []
        for r in self._col_reps:
            u1_col_reps.append(_O2_rep_to_U1(r))

        blocks = []
        block_irreps = []
        # generate Sz < 0 blocks
        # TODO

        # block 0 may not exist
        i0 = (self._block_irreps > 0).nonzero()[0][0]
        if i0 > 0:
            pro, pre, pco, pce = _get_b0_projectors(self._row_reps, self._col_reps)
            b0 = pro.T @ self._blocks[0] @ pco.T + pre.T @ self._blocks[1] @ pce.T
            blocks.append(b0)
            block_irreps.append(0)

        # add Sz > 0 blocks
        blocks.extend(self._blocks[i0:])
        block_irreps.extend(self._block_irreps[i0:])

        return U1_SymmetricTensor(u1_row_reps, u1_col_reps, blocks, block_irreps)

    def _permutate(self, row_axes, col_axes):
        # inefficient implementation: cast to U(1), permutate, then cast back to O(2)
        # TODO implement efficient specific permutate

        # construt new axes, conjugate if axis changes between row adnd column
        nrr = len(self._row_reps)
        row_reps = []
        for ax in row_axes:
            if ax < nrr:
                row_reps.append(self._row_reps[ax])
            else:
                row_reps.append(self.conjugate_representation(self._col_reps[ax - nrr]))
        col_reps = []
        for ax in col_axes:
            if ax < nrr:
                col_reps.append(self.conjugate_representation(self._row_reps[ax]))
            else:
                col_reps.append(self._col_reps[ax - nrr])

        tu1 = self.toU1()._permutate(row_axes, col_axes)
        return self.from_U1(tu1, row_reps, col_reps)

    def group_conjugated(self):
        # or should block -1 change sign?
        # should all odd blocks change sign?
        return self

    def check_blocks_fit_representations(self):
        assert self._block_irreps.size == self._nblocks
        assert len(self._blocks) == self._nblocks
        row_irreps = self.get_row_representation()
        col_irreps = self.get_column_representation()
        for (irr, b) in zip(self._block_irreps, self._blocks):
            nr = (row_irreps == irr).sum()
            nc = (col_irreps == irr).sum()
            assert nr > 0
            assert nc > 0
            assert b.shape == (nr, nc)
        return True
