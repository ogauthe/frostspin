import bisect
import operator
import functools

import numpy as np
import scipy.linalg as lg
import scipy.sparse as ssp
import numba

from groups.su2_representation import SU2_Representation


def get_projector(in1, in2, max_irrep=np.inf):
    # max_irrep cannot be set to None since irr3 loop depends on it
    out = in1 * in2
    out = out.truncated(max_irrep)
    shift3 = np.zeros(out.irreps[-1] + 1, dtype=int)
    n = 0
    row = []
    col = []
    data = []
    for i, irr3 in enumerate(out.irreps):
        shift3[irr3] = n  # indexed with IRREP, not index
        n += out.degen[i] * irr3
    cs1 = [0, *(in1.degen * in1.irreps).cumsum()]  # remember where to restart in in1
    cs2 = [0, *(in2.degen * in2.irreps).cumsum()]  # remember where to restart in in2
    for i1, irr1 in enumerate(in1.irreps):
        for i2, irr2 in enumerate(in2.irreps):
            d2 = in2.degen[i2]
            ar = np.arange(d2)
            sl2 = np.arange(cs2[i2], cs2[i2] + d2 * irr2)[:, None] * out.dim
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_irrep + 1), 2):
                p123 = SU2_Representation.elementary_projectors[irr1, irr2, irr3]
                sh = (irr1, d2, irr2, d2, irr3)
                temp = np.zeros(sh)
                temp[:, ar, :, ar] = p123
                temp = temp.reshape(irr1, d2 ** 2 * irr2 * irr3)
                row123, col123 = temp.nonzero()
                data123 = temp[row123, col123]
                shift1 = cs1[i1]
                for d1 in range(in1.degen[i1]):
                    full_col = (
                        sl2 + np.arange(shift3[irr3], shift3[irr3] + d2 * irr3)
                    ).ravel()
                    row.extend(shift1 + row123)
                    col.extend(full_col[col123])
                    data.extend(data123)
                    shift3[irr3] += d2 * irr3
                    shift1 += irr1
    sh = (in1.dim, in2.dim * out.dim)  # contract 1st leg in chained
    return ssp.csr_matrix((data, (row, col)), shape=sh)


def get_projector_chained(*rep_in, singlet_only=False):
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
        forwards.append(forwards[i - 1] * rep_in[i])
        backwards.append(backwards[i - 1] * rep_in[-i - 1])

    if singlet_only:
        # projection is made only on singlet. Remove irreps that wont fuse to 1.
        if forwards[-1].irreps[0] != 1:
            raise ValueError("No singlet in product")
        truncations = [1]
        forwards[-1] = forwards[-1].truncated(1)
        for i in range(n - 1):
            trunc = backwards[i].max_irrep
            forwards[-i - 2] = forwards[-i].truncated(trunc)
            truncations.append(trunc)
    else:
        truncations = [np.inf] * n

    proj = get_projector(forwards[0], rep_in[1], max_irrep=truncations[-2])
    for (f, rep, trunc) in zip(forwards[1:], rep_in[2:], reversed(truncations[:-2])):
        p = get_projector(f, rep, max_irrep=trunc)
        proj = proj.reshape(-1, p.shape[0]) @ p
    proj = proj.reshape(-1, forwards[-1].dim).tocsc()  # need to slice columns
    return proj


def construct_matrix_projector(rep_left_enum, rep_right_enum, conj_right=False):
    r"""
                singlet space
                /          \
               /            \
            prod_l        prod_r
             /               /
            /\              /\
           /\ \            /\ \
         rep_left        rep_right

    Returns:
    --------
    proj : (M, N) sparse matrix
        Projector on singlet, with N the singlet space dimension and M the Sz=0 input
        sector dimension.
    indices : (M,) integer ndarray
        Indices of proj rows in terms of dense matrix labelling.
    """
    repL = functools.reduce(operator.mul, rep_left_enum)
    repR = functools.reduce(operator.mul, rep_right_enum)
    dimLR = repL.dim * repR.dim
    projL = get_projector_chained(*rep_left_enum)
    projR = get_projector_chained(*rep_right_enum)
    if conj_right:  # same as conjugating input irrep, with smaller dimensions
        projR = projR @ ssp.csc_matrix(repR.get_conjugator())

    target = sorted(set(repL.irreps).intersection(repR.irreps))
    if not target:
        raise ValueError("Representations have no common irrep")
    repL = repL.truncated(target[-1])
    repR = repR.truncated(target[-1])
    if target != list(repL.irreps) or target != list(repR.irreps):  # TODO
        raise NotImplementedError("TODO: remove irreps that will not fuse")

    row = []
    col = []
    data = []
    shiftL = 0
    shiftR = 0
    shift_out = 0
    for i, irr in enumerate(target):
        degenR = repR.degen[i]
        matR = projR[:, shiftR : shiftR + degenR * irr].reshape(-1, irr)
        matR = matR.T.tocsr()
        sing_proj = ssp.csr_matrix(
            SU2_Representation.irrep(irr).get_conjugator() / np.sqrt(irr)
        )
        matR = sing_proj @ matR
        # it is not memory efficient to contract directly with the full matL: in csr,
        # indptr has size nrows, which would be dimL * degenL, much too large (saturates
        # memory). It also requires some sparse transpose. Using csc just puts the
        # problem on matR instead of matL. So to save memory, slice projL irrep by irrep
        # instead of taking all of them with degenL * irr. Slower but memory efficient.
        for j in range(repL.degen[i]):
            matLR = projL[:, shiftL : shiftL + irr].tocsr()  # avoid large indptr
            matLR = matLR @ matR
            matLR = matLR.tocoo().reshape(dimLR, degenR)  # force coo cast
            row.extend(matLR.row)
            col.extend(shift_out + matLR.col)
            data.extend(matLR.data)
            shiftL += irr
            shift_out += degenR
        shiftR += degenR * irr

    assert shift_out == repL.degen @ repR.degen
    full_proj = ssp.csr_matrix((data, (row, col)), shape=(dimLR, shift_out))
    return full_proj


def construct_transpose_matrix(representations, n_bra_leg1, n_bra_leg2, swap):
    r"""
    Construct isometry corresponding to change of tree structure of a SU(2) matrix.

                initial
                /      \
               /\      /\
              /\ \    / /\
              bra1    ket1
              |||      |||
             transposition
              ||||      ||
              bra2     ket2
             \/ / /     \  /
              \/ /       \/
               \/        /
                \       /
                  output

    Parameters
    ----------
    representations: enumerable of SU2_Representation
        List of SU(2) representation on which the matrix acts.
    n_bra_leg1: int
        Matrix to transpose has representations[:n_bra_leg1] as left leg and the others
        as right leg.
    n_bra_leg2: int
        Number of representations in left leg of transposed matrix.
    swap: enumerable of int
        Leg permutations, taking left and right legs in a row.
    """
    assert len(representations) == len(swap)
    sh1 = tuple(r.dim for r in representations)
    proj1 = construct_matrix_projector(
        representations[:n_bra_leg1], representations[n_bra_leg1:]
    )

    rep_bra2 = tuple(representations[i] for i in swap[:n_bra_leg2])
    rep_ket2 = tuple(representations[i] for i in swap[n_bra_leg2:])
    proj2 = construct_matrix_projector(rep_bra2, rep_ket2)

    # so, now we have initial shape projector and output shape projector. We need to
    # transpose rows to contract them. Since there is no bra/ket exchange, this can be
    # done by pure advanced slicing, without calling heavier sparse_transpose.
    sh2 = tuple(r.dim for r in rep_bra2) + tuple(r.dim for r in rep_ket2)
    strides1 = np.array((1,) + sh1[:0:-1]).cumprod()[::-1]
    strides2 = np.array((1,) + sh2[:0:-1]).cumprod()[::-1]
    nrows = (np.arange(proj1.shape[0])[:, None] // strides1 % sh1)[:, swap] @ strides2

    proj2 = proj2[nrows].T.tocsr()
    iso = proj2 @ proj1
    # tests show that construct_matrix_projector output has no numerical zeros
    # however iso may have more than 70% "non-zero" coeff being numerical zeros,
    # with several order of magnitude between them and real non-zeros.
    iso.data[np.abs(iso.data) < 1e-14] = 0
    iso.eliminate_zeros()
    iso = iso.sorted_indices()  # copy to get clean data array
    return iso


@numba.njit
def blocks_from_raw_data(degen_in, irreps_in, degen_out, irreps_out, data):
    i1 = 0
    i2 = 0
    blocks = []
    block_irreps = []
    k = 0
    while i1 < irreps_in.size and i2 < irreps_out.size:
        if irreps_in[i1] == irreps_out[i2]:
            sh = (degen_in[i1], degen_out[i2])
            m = data[k : k + sh[0] * sh[1]].reshape(sh) / np.sqrt(irreps_in[i1])
            blocks.append(m)
            k += m.size
            block_irreps.append(irreps_in[i1])
            i1 += 1
            i2 += 1
        elif irreps_in[i1] < irreps_out[i2]:
            i1 += 1
        else:
            i2 += 1
    return blocks, block_irreps


class SU2_Matrix:
    __array_priority__ = 15.0  # bypass ndarray.__mul__

    def __init__(self, blocks, block_irreps, left_rep, right_rep):
        # need block_irreps since some blocks may be zero
        assert len(blocks) == len(block_irreps)
        self._blocks = blocks
        self._block_irreps = block_irreps
        self._nblocks = len(blocks)
        self._left_rep = left_rep
        self._right_rep = right_rep

    @property
    def shape(self):
        return (self._left_rep.dim, self._right_rep.dim)

    @property
    def left_rep(self):
        return self._left_rep

    @property
    def right_rep(self):
        return self._right_rep

    @classmethod
    def from_raw_data(cls, data, rep_in, rep_out):
        blocks, block_irreps = blocks_from_raw_data(
            rep_in.degen, rep_in.irreps, rep_out.degen, rep_out.irreps, data
        )
        return cls(blocks, block_irreps, rep_in, rep_out)

    @classmethod
    def from_dense(cls, mat, rep_left_enum, rep_right_enum):
        prod_l = functools.reduce(operator.mul, rep_left_enum)
        prod_r = functools.reduce(operator.mul, rep_right_enum)
        proj = construct_matrix_projector(
            rep_left_enum, rep_right_enum, conj_right=True
        )
        data = proj.T @ mat.ravel()
        return cls.from_raw_data(data, prod_l, prod_r)

    def to_raw_data(self):
        # some blocks may be allowed by SU(2) in current matrix form but be zero and
        # missing in block_irreps (matrix created by matrix product). Still, data has to
        # include the corresponding zeros at the accurate position.
        shared, indL, indR = np.intersect1d(  # bruteforce numpy > clever python
            self._left_rep.irreps,
            self._right_rep.irreps,
            assume_unique=True,
            return_indices=True,
        )
        data = np.zeros(self._left_rep.degen[indL] @ self._right_rep.degen[indR])
        k = 0
        # hard to jit, need to call self._blocks[i] which may be heterogenous
        for i, irr in enumerate(shared):
            j = bisect.bisect_left(self._block_irreps, irr)
            if j < self._nblocks and self._block_irreps[j] == irr:
                b = self._blocks[j]
                data[k : k + b.size] = b.ravel() * np.sqrt(irr)
                k += b.size
            else:  # missing block
                k += self._left_rep.degen[indL[i]] * self._right_rep.degen[indR[i]]
        return data

    def toarray(self, rep_left_enum=None, rep_right_enum=None):
        if rep_left_enum is None:
            rep_left_enum = (self._left_rep,)
        if rep_right_enum is None:
            rep_right_enum = (self._right_rep,)
        proj = construct_matrix_projector(rep_left_enum, rep_right_enum)
        arr = proj @ self.to_raw_data()
        return arr.reshape(self.shape)

    def __repr__(self):
        s = "SU2_Matrix with irreps and shapes:\n"
        for irr, b in zip(self._block_irreps, self._blocks):
            s += f"{irr}: {b.shape}\n"
        return s

    def __mul__(self, x):
        y = np.atleast_2d(x)
        if y.size == 1:  # scalar multiplication
            blocks = [b * x for b in self._blocks]
        elif y.shape[0] == 1:  # diagonal weights applied on the right
            blocks = []
            k = 0
            for b in self._blocks:
                blocks.append(b * y[:, k : k + b.shape[1]])
                k += b.shape[1]
            if k != y.size:
                raise ValueError("Operand has non-compatible shape")
        elif y.shape[1] == 1:  # diagonal weights applied on the left
            blocks = []
            k = 0
            for b in self._blocks:
                blocks.append(b * y[k : k + b.shape[0]])
                k += b.shape[0]
            if k != y.size:
                raise ValueError("Operand has non-compatible shape")
        else:
            raise ValueError("Operand must be scalar or 1D vector")
        return SU2_Matrix(blocks, self._block_irreps, self._left_rep, self._right_rep)

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

    def __neg__(self):
        blocks = [-b for b in self._blocks]
        return SU2_Matrix(blocks, self._block_irreps, self._left_rep, self._right_rep)

    @property
    def T(self):
        blocks = [b.T for b in self._blocks]
        return SU2_Matrix(blocks, self._block_irreps, self._right_rep, self._left_rep)

    def norm(self):
        """
        Compute Frobenius norm.
        """
        n2 = 0.0
        for (irr, b) in zip(self._block_irreps, self._blocks):
            n2 += irr * lg.norm(b) ** 2
        return np.sqrt(n2)

    def __matmul__(self, other):
        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_irreps[i1] == other._block_irreps[i2]:
                blocks.append(self._blocks[i1] @ other._blocks[i2])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < other._block_irreps[i2]:
                i1 += 1
            else:
                i2 += 1
        return SU2_Matrix(blocks, block_irreps, self._left_rep, other._right_rep)

    def __add__(self, other):
        # not that left_rep and right_rep are product of input / output of rep. They may
        # correspond to different decompositions and addition would not be allowed for
        # dense tensors (different shapes) / meaningless for matrices.
        if self._left_rep != other._left_rep or self._right_rep != other._right_rep:
            raise ValueError("Matrices have non-compatible representations")
        blocks = []
        block_irreps = []
        i1, i2 = 0, 0
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_irreps[i1] == other._block_irreps[i2]:
                blocks.append(self._blocks[i1] + other._blocks[i2])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < other._block_irreps[i2]:
                blocks.append(self._block[i1])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
            else:
                blocks.append(other._block[i2])
                block_irreps.append(other._block_irreps[i2])
                i2 += 1
        return SU2_Matrix(blocks, block_irreps, self._left_rep, other._right_rep)

    def __sub__(self, other):
        return self + (-other)

    def expm(self):
        """
        Compute expm(self)
        """
        blocks = [lg.expm(b) for b in self._blocks]
        return SU2_Matrix(blocks, self._block_irreps, self._left_rep, self._right_rep)

    def svd(self, cut=None, rcutoff=0.0):
        """
        Compute block-wise SVD of self and keep only cut largest singular values. Do not
        truncate if cut is not provided. Keep only values larger than rcutoff * max(sv).
        """
        block_u = [None] * self._nblocks
        block_s = [None] * self._nblocks
        block_v = [None] * self._nblocks
        block_max_vals = np.empty(self._nblocks)
        for bi, b in enumerate(self._blocks):
            block_u[bi], block_s[bi], block_v[bi] = lg.svd(
                b, full_matrices=False, check_finite=False
            )
            block_max_vals[bi] = block_s[bi][0]

        cutoff = block_max_vals.max() * rcutoff  # cannot be set before 1st loop
        block_cuts = [0] * self._nblocks
        if cut is None:
            if rcutoff > 0.0:  # remove values smaller than cutoff
                for bi, bs in enumerate(block_s):
                    keep = (bs > cutoff).nonzero()[0]
                    if keep.size:
                        block_cuts[bi] = keep[-1] + 1
            else:
                block_cuts = [b.size for b in block_s]
        else:  # Assume number of blocks is small, block_max_val is never sorted
            k = 0  # and elements are compared at each iteration
            while k < cut:
                bi = block_max_vals.argmax()
                if block_max_vals[bi] < cutoff:
                    break
                block_cuts[bi] += 1
                if block_cuts[bi] < block_s[bi].size:
                    block_max_vals[bi] = block_s[bi][block_cuts[bi]]
                else:
                    block_max_vals[bi] = -1.0  # in case cutoff = 0
                k += 1

        s = []
        for bi in reversed(range(self._nblocks)):  # reversed to del
            if block_cuts[bi]:
                block_u[bi] = block_u[bi][:, : block_cuts[bi]]
                s.extend(block_s[bi][: block_cuts[bi]][::-1])
                block_v[bi] = block_v[bi][: block_cuts[bi]]
            else:  # do not keep empty matrices
                del block_u[bi]
                del block_v[bi]

        mid_rep = SU2_Representation(block_cuts, self._block_irreps)
        U = SU2_Matrix(block_u, mid_rep.irreps, self._left_rep, mid_rep)
        V = SU2_Matrix(block_v, mid_rep.irreps, mid_rep, self._right_rep)
        s = np.array(s[::-1])
        return U, s, V, mid_rep
