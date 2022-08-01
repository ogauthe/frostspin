import numpy as np
import scipy.sparse as sp
import scipy.linalg as lg
import numba

from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor


def custom_sparse_product(p1, b, p2):
    """
    Compute p1 @ b @ p2, where b is a dense block and p1 and p2 are sparse matrices
    """
    p1r = p1.tocsr()
    p2c = p2.tocsc()
    return _numba_custom_sparse(
        p1r.shape[0],
        p1r.indptr,
        p1r.indices,
        p1r.data,
        b,
        p2c.shape[1],
        p2c.indptr,
        p2c.indices,
        p2c.data,
    )


@numba.njit(parallel=True)
def _numba_custom_sparse(
    nrow1, row_pointer1, col_indices1, nnz1, b, ncol2, col_pointer2, row_indices2, nnz2
):
    Y = np.empty((nrow1, ncol2), dtype=b.dtype)
    for i in numba.prange(nrow1):
        for j in numba.prange(ncol2):
            s = 0.0
            for m in numba.prange(row_pointer1[i], row_pointer1[i + 1]):
                for n in numba.prange(col_pointer2[j], col_pointer2[j + 1]):
                    s += nnz1[m] * b[col_indices1[m], row_indices2[n]] * nnz2[n]
            Y[i, j] = s
    return Y


@numba.njit
def _numba_combine_O2(*reps):
    degen, irreps = reps[0]
    for r in reps[1:]:
        degen, irreps = _numba_elementary_combine_O2(degen, irreps, r[0], r[1])
    return np.concatenate((degen, irreps)).reshape(2, -1)


@numba.njit
def _numba_elementary_combine_O2(degen1, irreps1, degen2, irreps2):
    # 0o x 0o = 0e
    # 0e x 0e = 0e
    # 0e x 0o = 0o
    # 0o x 0e = 0o
    # 0o x n = n
    # 0e x n = n
    # n x n = 0o + 0e + 2n
    # n x m = |n-m| + (n+m)
    nmax = max(irreps1[-1], 0) + max(irreps2[-1], 0)
    degen = np.zeros((nmax + 2,), dtype=np.int64)

    for (d1, irr1) in zip(degen1, irreps1):
        for (d2, irr2) in zip(degen2, irreps2):
            d = d1 * d2
            if irr1 < 1:
                if irr2 < 1:
                    degen[(irr1 + irr2 + 1) % 2] += d
                else:
                    degen[irr2 + 1] += d
            else:
                if irr2 < 1:
                    degen[irr1 + 1] += d
                elif irr1 == irr2:
                    degen[0] += d
                    degen[1] += d
                    degen[2 * irr1 + 1] += d
                else:
                    degen[irr1 + irr2 + 1] += d
                    degen[abs(irr1 - irr2) + 1] += d

    irreps = np.arange(-1, nmax + 1)
    nnz = degen.nonzero()[0]
    return degen[nnz], irreps[nnz]


@numba.njit
def _numba_O2_representation_dimension(r):
    return (r[0] * ((r[1] > 0).astype(np.int64) + 1)).sum()


@numba.njit
def _numba_O2_rep_to_U1(r):
    """
    Map O(2) representation to U(1) representation
    O(2) representations starts with irreps 0odd, 0even which map to U(1) 0
    then O(2) irreps merge +n and -n U(1) for n>0
    maps to U(1) irrep +n and -n, with contiguous +n and -n sectors

    ex: [[1, 2, 1, 2], [-1, 0, 1, 2]] is mapped to [0, 0, 0, 1, -1, 2, 2, -2, -2]
    """
    # ordering is a bit more complex than consecutive pairs (n,-n)
    # hope to improve perf with these consecutive U(1) sectors
    ru1 = np.empty((_numba_O2_representation_dimension(r),), dtype=np.int8)
    i = 0
    k = 0
    if r[1, 0] == -1:
        ru1[: r[0, 0]] = 0
        k += r[0, 0]
        i += 1
    if r[1, i] == 0:
        ru1[k : k + r[0, i]] = 0
        k += r[0, i]
        i += 1
    for j in range(i, r.shape[1]):
        ru1[k : k + r[0, j]] = r[1, j]
        k += r[0, j]
        ru1[k : k + r[0, j]] = -r[1, j]
        k += r[0, j]
    return ru1


@numba.njit
def _numba_get_coo_proj(signs, so):
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


@numba.njit
def _numba_get_reflection_perm_sign(rep):
    """
    Construct basis permutation mapping vectors into their reflected form.

    Parameters
    ----------
    rep: O(2) representation
    """
    d = _numba_O2_representation_dimension(rep)
    perm = np.arange(d)
    sign = np.ones((d,), dtype=np.int8)
    i1 = np.searchsorted(rep[1], 1)  # slice containing 0o and 0e if they exist
    if rep[1, 0] == -1:
        sign[: rep[0, 0]] = -1
    k = rep[0, :i1].sum()
    for i in range(i1, rep.shape[1]):
        degen = rep[0, i]
        perm[k : k + degen] = np.arange(k + degen, k + 2 * degen)
        perm[k + degen : k + 2 * degen] = np.arange(k, k + degen)
        if rep[1, i] % 2:  # -1 sign for Sz<0 to Sz>0
            sign[k + degen : k + 2 * degen] = -1
        k += 2 * degen
    return perm, sign


@numba.njit
def _numba_get_swapped(indices, strides, reps):
    indices_b = np.zeros(indices.shape[0], dtype=np.int64)
    signs = np.ones((indices.shape[0],), dtype=np.int8)
    for i, r in enumerate(reps):
        p, s = _numba_get_reflection_perm_sign(r)
        indices_b += p[indices[:, i]] * strides[i]  # map all indices to spin reversed
        signs *= s[indices[:, i]]
    return indices_b, signs


def _get_b0_projectors(row_reps, col_reps, signature):
    """
    Construct 0odd and 0even projectors. They allow to decompose U(1) Sz=0 sector into
    0odd (aka -1) and 0even (aka 0) sectors.
    """

    # TODO have only rsig, rso, csign, cso as input
    # reuse these for all Sz to generate -Sz block
    nrr = len(row_reps)
    shr = np.array([_numba_O2_representation_dimension(r) for r in row_reps])
    row_cp = np.array([1, *shr[-1:0:-1]]).cumprod()[::-1]
    u1_row_reps = tuple(_numba_O2_rep_to_U1(r) for r in row_reps)
    u1_combined = U1_SymmetricTensor.combine_representations(
        u1_row_reps, signature[:nrr]
    )
    rso = ((u1_combined == 0).nonzero()[0] // row_cp[:, None]).T % shr
    rso, rsign = _numba_get_swapped(rso, row_cp, tuple(row_reps))
    rso = rso.argsort()  # imposed sorted block indices in U(1) => argsort
    ie, ecoeff, erows, ecols, io, ocoeff, orows, ocols = _numba_get_coo_proj(rsign, rso)
    pre = sp.coo_matrix((ecoeff, (erows, ecols)), shape=(ie, rso.size))
    pro = sp.coo_matrix((ocoeff, (orows, ocols)), shape=(io, rso.size))

    # same operation for columns
    shc = [_numba_O2_representation_dimension(r) for r in col_reps]
    col_cp = np.array([1, *shc[-1:0:-1]]).cumprod()[::-1]
    u1_col_reps = tuple(_numba_O2_rep_to_U1(r) for r in col_reps)
    u1_combined = U1_SymmetricTensor.combine_representations(
        u1_col_reps, ~signature[nrr:]
    )
    cso = ((u1_combined == 0).nonzero()[0] // col_cp[:, None]).T % shc
    cso, csign = _numba_get_swapped(cso, col_cp, tuple(col_reps))
    cso = cso.argsort()  # imposed sorted block indices in U(1) => argsort
    ie, ecoeff, erows, ecols, io, ocoeff, orows, ocols = _numba_get_coo_proj(csign, cso)
    pce = sp.coo_matrix((ecoeff, (ecols, erows)), shape=(cso.size, ie))
    pco = sp.coo_matrix((ocoeff, (ocols, orows)), shape=(cso.size, io))

    return pro, pre, pco, pce


@numba.njit(parallel=True)
def _numba_generate_block(rso, cso, bsigns):
    b = np.empty((rso.size, cso.size), dtype=bsigns.dtype)
    for i in numba.prange(rso.size):
        for j in numba.prange(cso.size):
            b[rso[i], cso[j]] = bsigns[i, j]
    return b


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

    @classmethod
    @property
    def singlet(cls):
        return np.array([[1], [0]])

    @staticmethod
    def representation_dimension(rep):
        return _numba_O2_representation_dimension(rep)

    @staticmethod
    def irrep_dimension(irrep):
        return 1 + (irrep > 0)

    @staticmethod
    def combine_representations(reps, signature):
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
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        # not fully efficient: 1st construct U(1) symmetric tensor to get abelian blocks
        # then split sector 0 into 0even and 0odd
        # discard sectors with n < 0
        # keep sectors with n > 0 as they are
        u1_row_reps = tuple(_numba_O2_rep_to_U1(r) for r in row_reps)
        u1_col_reps = tuple(_numba_O2_rep_to_U1(r) for r in col_reps)
        tu1 = U1_SymmetricTensor.from_array(arr, u1_row_reps, u1_col_reps, signature)
        to2 = cls.from_U1(tu1, row_reps, col_reps)
        assert abs(to2.norm() - lg.norm(arr)) <= 1e-14 * lg.norm(
            arr
        ), "norm is not conserved in O2_SymmetricTensor cast"
        return to2

    @classmethod
    def from_U1(cls, tu1, row_reps, col_reps):
        """
        Assume tu1 has O(2) symmetry and its irreps are sorted according to O(2) rules.

        No check is made on tensor norm at the end, so that U(1) tensor with only
        Sz >= 0 blocks may be used to initialize O(2) tensor. Allows to use partial U(1)
        tensor as intermediate step to construct O(2) without the additional cost of
        constructing Sz < 0 blocks and benefit from Sz = 0 block splitting.
        """

        # may remove these checks to be able to use U(1) with combined rows and columns
        # when tu1 is a temporary object
        # just need signature as input instead of tu1.signature
        assert tu1._nrr == len(row_reps)
        assert all(
            (_numba_O2_rep_to_U1(r) == tu1.row_reps[i]).all()
            for i, r in enumerate(row_reps)
        )
        assert len(tu1.col_reps) == len(col_reps)
        assert all(
            (_numba_O2_rep_to_U1(r) == tu1.col_reps[i]).all()
            for i, r in enumerate(col_reps)
        )

        blocks = []
        block_irreps = []
        i0 = tu1.block_irreps.searchsorted(0)
        if tu1.nblocks > i0 and tu1.block_irreps[i0] == 0:
            i1 = i0 + 1
            pro, pre, pco, pce = _get_b0_projectors(row_reps, col_reps, tu1.signature)
            if pro.shape[0] > 0 and pco.shape[1] > 0:
                block_irreps.append(-1)
                b0o = custom_sparse_product(pro, tu1.blocks[i0], pco)
                blocks.append(b0o)
            if pre.shape[0] > 0 and pce.shape[1] > 0:
                block_irreps.append(0)
                b0e = custom_sparse_product(pre, tu1.blocks[i0], pce)
                blocks.append(b0e)
            assert abs(
                np.sqrt(sum(lg.norm(b) ** 2 for b in blocks)) - lg.norm(tu1.blocks[i0])
            ) <= 1e-14 * lg.norm(tu1.blocks[i0])
        else:
            i1 = i0
        block_irreps.extend(tu1.block_irreps[i1:])
        blocks.extend(tu1.blocks[i1:])
        return cls(row_reps, col_reps, blocks, block_irreps, tu1.signature)

    def toarray(self, as_matrix=False):
        return self.toU1().toarray(as_matrix)

    def toabelian(self):
        return self.toU1()

    def _generate_neg_sz_blocks(self):
        u1_row_reps = [None] * self._nrr
        rmaps = [None] * self._nrr
        rsigns = [None] * self._nrr
        for i, r in enumerate(self._row_reps):
            u1_row_reps[i] = _numba_O2_rep_to_U1(r)
            rmaps[i], rsigns[i] = _numba_get_reflection_perm_sign(r)

        shr = np.array(self.shape[: self._nrr])
        row_cp = np.array([1, *shr[-1:0:-1]]).cumprod()[::-1]
        u1_combined_row = U1_SymmetricTensor.combine_representations(
            u1_row_reps, self._signature[: self._nrr]
        )

        ncr = len(self._col_reps)
        u1_col_reps = [None] * ncr
        cmaps = [None] * ncr
        csigns = [None] * ncr
        for i, r in enumerate(self._col_reps):
            u1_col_reps[i] = _numba_O2_rep_to_U1(r)
            cmaps[i], csigns[i] = _numba_get_reflection_perm_sign(r)

        shc = np.array(self.shape[self._nrr :])
        col_cp = np.array([1, *shc[-1:0:-1]]).cumprod()[::-1]
        u1_combined_col = U1_SymmetricTensor.combine_representations(
            u1_col_reps, ~self._signature[self._nrr :]
        )

        blocks = []
        isz = self._nblocks - 1
        while isz > -1 and self._block_irreps[isz] > 0:
            sz = self._block_irreps[isz]
            # could factorize Sz-reflection, but implies doing operation for all Sz
            # better to map to Sz-reflected inside loop, only for Sz<0?
            # or numpy will make things faster?
            rsz_mat = (u1_combined_row == sz).nonzero()[0]  # find Sz states
            rsz_t = (rsz_mat // row_cp[:, None]).T % shr  # multi-index form
            rszb_mat = np.zeros((rsz_mat.size,), dtype=int)
            rsign = np.ones((rsz_mat.size,), dtype=np.int8)
            for i, r in enumerate(self._row_reps):
                rszb_mat += rmaps[i][rsz_t[:, i]] * row_cp[i]  # map to spin reversed
                rsign *= rsigns[i][rsz_t[:, i]]

            csz_mat = (u1_combined_col == sz).nonzero()[0]  # find Sz states
            csz_t = (csz_mat // col_cp[:, None]).T % shc  # multi-index form
            cszb_mat = np.zeros((csz_mat.size,), dtype=int)
            csign = np.ones((csz_mat.size,), dtype=np.int8)
            for i, r in enumerate(self._col_reps):
                cszb_mat += cmaps[i][csz_t[:, i]] * col_cp[i]  # map to spin reversed
                csign *= csigns[i][csz_t[:, i]]

            rso = rszb_mat.argsort().argsort()
            cso = cszb_mat.argsort().argsort()
            bsign = rsign[:, None] * self._blocks[isz] * csign
            b = _numba_generate_block(rso, cso, bsign)
            blocks.append(b)
            isz -= 1

        return blocks

    def toO2(self):
        return self

    def toU1(self):
        # Sz < 0 blocks
        blocks = self._generate_neg_sz_blocks()
        block_irreps = -self._block_irreps[::-1]

        # Sz = 0 blocks (may not exist)
        if self._block_irreps[0] < 1:
            pro, pre, pco, pce = _get_b0_projectors(
                self._row_reps, self._col_reps, self._signature
            )
            # try to reduce number of copies due to F ordering
            if self._block_irreps[0] == 0:  # no 0o block
                i1 = 1
                b0 = custom_sparse_product(pre.T, self._blocks[0], pce.T)
                block_irreps = np.hstack((block_irreps, self._block_irreps[1:]))
            elif self.nblocks > 1 and self._block_irreps[1] > 0:  # no 0e block
                i1 = 1
                b0 = custom_sparse_product(pro.T, self._blocks[0], pco.T)
                block_irreps[-1] = 0
                block_irreps = np.hstack((block_irreps, self._block_irreps[1:]))
            else:
                i1 = 2
                b0 = custom_sparse_product(
                    pro.T, self._blocks[0], pco.T
                ) + custom_sparse_product(pre.T, self._blocks[1], pce.T)
                block_irreps = np.hstack((block_irreps[:-1], self._block_irreps[2:]))
            blocks.append(b0)
        else:
            i1 = 0
            block_irreps = np.hstack((block_irreps, self._block_irreps))

        # Sz > 0 blocks
        blocks.extend(self._blocks[i1:])

        u1_row_reps = tuple(_numba_O2_rep_to_U1(r) for r in self._row_reps)
        u1_col_reps = tuple(_numba_O2_rep_to_U1(r) for r in self._col_reps)
        tu1 = U1_SymmetricTensor(
            u1_row_reps, u1_col_reps, blocks, block_irreps, self._signature
        )
        assert abs(tu1.norm() - self.norm()) <= 1e-14 * self.norm()
        return tu1

    def permutate(self, row_axes, col_axes):
        assert sorted(row_axes + col_axes) == list(range(self._ndim))

        # return early for identity or matrix transpose
        if row_axes == tuple(range(self._nrr)) and col_axes == tuple(
            range(self._nrr, self._ndim)
        ):
            return self
        if row_axes == tuple(range(self._nrr, self._ndim)) and col_axes == tuple(
            range(self._nrr)
        ):
            return self.T

        # inefficient implementation: cast to U(1), permutate, then cast back to O(2)
        # TODO implement efficient specific permutate
        reps = []
        for ax in row_axes + col_axes:
            if ax < self._nrr:
                reps.append(self._row_reps[ax])
            else:
                reps.append(self._col_reps[ax - self._nrr])
        tu1 = self.toU1().permutate(row_axes, col_axes)
        tp = self.from_U1(tu1, reps[: len(row_axes)], reps[len(row_axes) :])
        assert (
            abs(tp.norm() - self.norm()) <= 1e-14 * self.norm()
        ), "permutate changes norm"
        return tp

    @property
    def T(self):
        b_neg = tuple(b.T for b in reversed(self._generate_neg_sz_blocks()))
        b0 = tuple(b.T for b in self._blocks[: self._block_irreps.searchsorted(1)])
        blocks = b0 + b_neg
        s = self._signature[np.arange(-self._ndim + self._nrr, self._nrr) % self._ndim]
        return type(self)(self._col_reps, self._row_reps, blocks, self._block_irreps, s)

    def group_conjugated(self):
        signature = ~self._signature
        blocks = tuple(self._generate_neg_sz_blocks()[::-1])
        blocks = self._blocks[: self._block_irreps.searchsorted(1)] + blocks
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, signature
        )

    def update_signature(self, sign_update):
        # same as abelian case, bending an index to the left or to the right makes no
        # difference, signs can be ignored.
        up = np.asarray(sign_update, dtype=bool)
        assert up.shape == (self._ndim,)
        row_reps = list(self._row_reps)
        col_reps = list(self._col_reps)
        for i in up.nonzero()[0]:
            if i < self._nrr:
                row_reps[i] = self.conjugate_representation(row_reps[i])
            else:
                j = i - self._nrr
                col_reps[j] = self.conjugate_representation(col_reps[j])
        self._row_reps = tuple(row_reps)
        self._col_reps = tuple(col_reps)
        self._signature = self._signature ^ up
        assert self.check_blocks_fit_representations()

    def check_blocks_fit_representations(self):
        assert self._block_irreps.size == self._nblocks
        assert len(self._blocks) == self._nblocks
        row_rep = self.get_row_representation()
        col_rep = self.get_column_representation()
        r_indices = row_rep[1].searchsorted(self._block_irreps)
        c_indices = col_rep[1].searchsorted(self._block_irreps)
        assert (row_rep[1, r_indices] == self._block_irreps).all()
        assert (col_rep[1, c_indices] == self._block_irreps).all()
        for bi in range(self._nblocks):
            nr = row_rep[0, r_indices[bi]]
            nc = col_rep[0, c_indices[bi]]
            assert nr > 0
            assert nc > 0
            assert self._blocks[bi].shape == (nr, nc)
        return True

    def merge_legs(self, i1, i2):
        return NotImplementedError("To do!")
