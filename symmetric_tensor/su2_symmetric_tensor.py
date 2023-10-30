import numpy as np
import scipy.sparse as ssp
import scipy.linalg as lg
import numba

from .lie_group_symmetric_tensor import LieGroupSymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor
from .o2_symmetric_tensor import O2_SymmetricTensor
from groups.su2_representation import SU2_Representation  # TODO remove me

_singlet = np.array([[1], [1]])


def _get_projector(rep1, rep2, s1, s2, max_irrep=2**30):
    """
    Construct Clebsch-Gordan fusion tensor for representations rep1 and rep2 with
    signatures s1 and s2.

    Parameters
    ----------
    rep1 : 2D int array
        Left incoming SU(2) representation to fuse.
    rep2 : 2D int array
        Right incoming SU(2) representation to fuse.
    s1 : bool
        Signature for rep1.
    s2 : bool
        Signature for rep2.
    max_irrep : int
        Dimension of maximal irrep to consider in the product. Irrep larger than
        max_irrep will be truncated. Default is 2**30, i.e. no truncation.

    Returns
    -------
    ret : csr_array
        CG projector fusing rep1 and rep2 on sum of irreps, truncated up to max_irrep.
        It has a 2D shape (dim_rep1, dim_rep2 * dim_out), where dim_out may be smaller
        than array dim_rep1 * dim_rep2 if truncation occured.

    Notes
    -----
        The output matrix hides a structure
        (degen1, irrep1, degen2, irrep2, degen1, degen2, irrep3).
        This is not exactly a tensor but corresponds to how row and columns relates to
        degeneracies and irreducible representations.
    """
    degen, irreps = _numba_elementary_combine_SU2(rep1[0], rep1[1], rep2[0], rep2[1])
    trunc = irreps.searchsorted(max_irrep + 1)
    degen = degen[:trunc]
    irreps = irreps[:trunc]
    out_dim = degen @ irreps

    # construct sparse matrix from row, col and data lists
    row = []
    col = []
    data = []

    # shift in sparse matrix indices corresponding to different output irreps
    shift3 = np.zeros((irreps[-1] + 1,), dtype=int)
    n = 0
    for d3, irr3 in zip(degen, irreps):
        shift3[irr3] = n  # indexed with IRREP, not index
        n += d3 * irr3

    # shift for different input irreps
    cs1 = [0, *(rep1[0] * rep1[1]).cumsum()]  # remember where to restart in rep1
    cs2 = [0, *(rep2[0] * rep2[1]).cumsum()]  # remember where to restart in rep2

    for i1, irr1 in enumerate(rep1[1]):
        # Sz-reversal signs for irrep1
        diag1 = (np.arange(irr1 % 2, irr1 + irr1 % 2) % 2 * 2 - 1)[:, None, None]

        for i2, irr2 in enumerate(rep2[1]):
            # Sz-reversal signs for irrep2
            diag2 = (np.arange(irr2 % 2, irr2 + irr2 % 2) % 2 * 2 - 1)[None, :, None]

            d2 = rep2[0, i2]
            ar = np.arange(d2)
            sl2 = np.arange(cs2[i2], cs2[i2] + d2 * irr2)[:, None] * out_dim
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_irrep + 1), 2):
                p123 = SU2_Representation.elementary_projectors[irr1, irr2, irr3]
                # apply spin-reversal operator according to signatures
                if s1:
                    p123 = p123[::-1] * diag1
                if s2:
                    p123 = p123[:, ::-1] * diag2

                # broadcast elementary projector for degen2
                sh = (irr1, d2, irr2, d2, irr3)
                temp = np.zeros(sh)
                temp[:, ar, :, ar] = p123
                temp = temp.reshape(irr1, d2 * irr2 * d2 * irr3)  # reshape as matrix

                row123, col123 = temp.nonzero()  # get non-zero coeff position
                data123 = temp[row123, col123]  # get non-zero coeff as a 1D array

                shift1 = cs1[i1]
                for d1 in range(rep1[0, i1]):  # broadcast for degen1
                    full_col = (sl2 + shift3[irr3] + np.arange(d2 * irr3)).ravel()
                    row.extend(shift1 + row123)
                    col.extend(full_col[col123])
                    data.extend(data123)
                    shift3[irr3] += d2 * irr3
                    shift1 += irr1

    # construct 2D sparse array merging rep2 with out
    dim_rep1 = rep1[0] @ rep1[1]
    dim_rep2 = rep2[0] @ rep2[1]
    sh = (dim_rep1, dim_rep2 * out_dim)  # contract 1st leg in chained
    ret = ssp.csr_array((data, (row, col)), shape=sh)
    return ret


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
    for d1, irr1 in zip(degen1, irreps1):
        for d2, irr2 in zip(degen2, irreps2):
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
    def conjugate_irrep(irr):
        return irr

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
    _isometry_dic = {}

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
        full_proj = ssp.csr_array((data, (row, col)), shape=(dimLR, shift_out))
        return full_proj

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    def group_conjugated(self):
        # .T makes use of precomputed isometries and .H is costless
        ret = self.T.H
        ret._blocks = tuple(b.conj() for b in ret._blocks)
        return ret

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
                    m = ssp.coo_array((coeff, (inds, inds)), shape=(d, d))
                else:
                    m = ssp.eye(d, format="coo")
                diag = ssp.kron(diag, m)

            # could be stored in unitary_dic if too expensive, expect uncommon operation
            diag = diag.todia()  # we know the matrix is diagonal
            raw_data = p.T @ (diag @ (p @ self._to_raw_data()))
            blocks, block_irreps = self._blocks_from_raw_data(
                raw_data,
                self.get_row_representation(),
                self.get_column_representation(),
            )
            self._blocks = blocks
            self._block_irreps = block_irreps

        self._signature = self._signature ^ up.astype(bool)

    def toSU2(self):
        return self

    def toabelian(self):
        return self.toU1()

    def toU1(self):
        # efficient cast to U(1): project directly raw data to U(1) blocks
        # 1) construct U(1) representations
        reps = []
        for r in self._row_reps + self._col_reps:
            sz = np.empty((r[0] @ r[1],), dtype=np.int8)
            k = 0
            for d, irr in zip(r[0], r[1]):
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
        raw = self._to_raw_data()
        for sz in block_irreps:
            ri = (row_irreps == sz).nonzero()[0]
            ci = (col_irreps == sz).nonzero()[0]
            inds = ncols * ri[:, None] + ci
            b = (proj[inds.ravel()] @ raw).reshape(inds.shape)
            blocks.append(b)

        assert (
            abs(np.sqrt(sum(lg.norm(b) ** 2 for b in blocks)) - self.norm())
            <= 1e-13 * self.norm()
        )
        return U1_SymmetricTensor(
            reps[: self._nrr], reps[self._nrr :], blocks, block_irreps, self._signature
        )

    def toO2(self):
        """
        WARNING: this method alters the dense tensors by swapping indices inside legs
        and adding some diagonal -1 signs on every legs. This does not matter once legs
        are contracted.
        """
        # When casting to U(1), O(2) has different irrep ordering conventions:
        # here for SU(2), each spin appears contiguously with all its Sz value
        # For O(2), dim 2 irreps are mixed to get a contiguous +n sector and a
        # contiguous -n sector
        # e.g. for 2 spins 1, SU(2).toU1() gives Sz = [2,0,-2,2,0,-2]
        # while for O(2), this gives 2 0odd and 2 irreps 2 with Sz = [0,0,2,2,-2,-2]
        # hence some permutation is needed for every leg.

        # Furthermore, SU(2) follows standard spin-reversal convention, with vector |Sz>
        # being map to +/-|-Sz> depending on irrep.
        # For O(2), we impose even Z|Sz> -> |-Sz> for integer Sz (even with factor 2)
        # Z|Sz> -> +|-Sz> for Sz > 0 and Z|Sz> -> -|-Sz> for Sz < 0 for half interger Sz
        # This means starting from spin 3/2, some basis vector must be redefined with a
        # -1 sign, which introduces signs in the tensor coefficients.

        # reuse U(1) code in spirit
        # 1) construct U(1) and O(2) representations
        swaps = []  # SU(2) and O(2) have different ordering convention: swap needed
        u1_reps = []
        o2_reps = []
        signs = []  # change vector signs to fit O(2) signs conventions
        for r in self._row_reps + self._col_reps:
            sz_rep = np.empty((r[0] @ r[1],), dtype=np.int8)
            sz_rep_o2 = np.empty(sz_rep.shape, dtype=np.int8)
            k = 0
            signs_rep = np.empty(sz_rep.shape, dtype=bool)
            for d, irr in zip(r[0], r[1]):
                sz_irr = np.arange(irr - 1, -irr - 1, -2, dtype=np.int8)
                sz_rep[k : k + d * irr].reshape(d, irr)[:] = sz_irr
                sz_irr_o2 = np.abs(sz_irr)
                signs_irr = np.zeros(sz_irr.shape, dtype=bool)
                if irr % 2:
                    signs_irr[(sz_irr % 4 == (irr + 1) % 4) & (sz_irr < 0)] = True
                    if irr % 4 == 3:  # odd integer spin -> Sz=0 is odd
                        sz_irr_o2[irr // 2] = -1
                else:
                    signs_irr[(sz_irr % 4 == (irr - 1) % 4) & (sz_irr < 0)] = True
                sz_rep_o2[k : k + d * irr].reshape(d, irr)[:] = sz_irr_o2
                signs_rep[k : k + d * irr].reshape(d, irr)[:] = signs_irr
                k += d * irr
            swap_rep = (-sz_rep).argsort(kind="stable")
            swap_rep = swap_rep[sz_rep_o2[swap_rep].argsort(kind="stable")]
            u1_reps.append(sz_rep[swap_rep])  # swap U(1) indices to O(2) format
            signs.append(signs_rep)
            swaps.append(swap_rep)  # store swap
            irreps, degen = np.unique(sz_rep_o2, return_counts=True)
            degen[irreps > 0] //= 2
            o2_reps.append(np.array([degen, irreps]))

        row_signs = signs[0]
        for i in range(1, self._nrr):
            row_signs = (row_signs[:, None] ^ signs[i]).ravel()
        col_signs = signs[self._nrr]
        for i in range(self._nrr, self._ndim):
            col_signs = (col_signs[:, None] ^ signs[i]).ravel()

        # 2) combine into row and column
        row_irreps = U1_SymmetricTensor.combine_representations(
            u1_reps[: self._nrr], self._signature[: self._nrr]
        )
        col_irreps = U1_SymmetricTensor.combine_representations(
            u1_reps[self._nrr :], ~self._signature[self._nrr :]
        )
        ncols = col_irreps.size

        # 3) find non-empty U(1) blocks directly from current block_irreps
        sze = (self._block_irreps % 2).nonzero()[0]  # integer spins
        szo = ((self._block_irreps + 1) % 2).nonzero()[0]  # half interger spins
        block_irreps = []
        if sze.size:
            sze_maxp1 = self._block_irreps[sze[-1]]
            block_irreps.extend(np.arange(0, sze_maxp1, 2))  # remove Sz < 0
        if szo.size:
            szo_maxp1 = self._block_irreps[szo[-1]]
            block_irreps.extend(np.arange(1, szo_maxp1, 2))  # remove Sz < 0
        block_irreps = np.sort(block_irreps)

        # 4) construct U(1) blocks by slicing Clebsh-Gordon projector
        # need to include swapping
        shr = np.array(self.shape[: self._nrr])
        row_cp = np.array([1, *shr[-1:0:-1]]).cumprod()[::-1]
        shc = np.array(self.shape[self._nrr :])
        col_cp = np.array([1, *shc[-1:0:-1]]).cumprod()[::-1]

        blocks = []
        proj = self.construct_matrix_projector(
            self._row_reps, self._col_reps, self._signature
        )
        raw = self._to_raw_data()
        for sz in block_irreps:
            rsz_mat = (row_irreps == sz).nonzero()[0]
            rsz_t = (rsz_mat // row_cp[:, None]).T % shr  # multi-index form
            rsz_mat[:] = 0  # unswapped form
            for i, r in enumerate(self._row_reps):
                rsz_mat += swaps[i][rsz_t[:, i]] * row_cp[i]  # map to unswapped Sz

            csz_mat = (col_irreps == sz).nonzero()[0]
            csz_t = (csz_mat // col_cp[:, None]).T % shc  # multi-index form
            csz_mat[:] = 0
            for i, r in enumerate(self._col_reps):
                csz_mat += swaps[i + self._nrr][csz_t[:, i]] * col_cp[i]

            # rsz_mat and csz_mat contains the same indices as without the swap, however
            # they are swapped according to axis-wise swap imposed by O(2) irreps order
            inds = ncols * rsz_mat[:, None] + csz_mat
            b = (proj[inds.ravel()] @ raw).reshape(inds.shape)
            b = (1 - 2 * row_signs[rsz_mat, None]) * b * (1 - 2 * col_signs[csz_mat])
            blocks.append(b)

        tu1 = U1_SymmetricTensor(  # blocks Sz<0 are missing (will not be read)
            u1_reps[: self._nrr],
            u1_reps[self._nrr :],
            blocks,
            block_irreps,
            self._signature,
        )
        to2 = O2_SymmetricTensor.from_U1(
            tu1, o2_reps[: self._nrr], o2_reps[self._nrr :]
        )
        assert abs(to2.norm() - self.norm()) <= 1e-13 * self.norm()
        return to2
