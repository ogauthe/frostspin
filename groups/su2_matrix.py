import bisect
import operator
import functools

import numpy as np
import scipy.linalg as lg

from groups.toolsU1 import combine_colors
from groups.block_matrix_U1 import BlockMatrixU1
from groups.su2_representation import SU2_Representation


def get_projector(in1, in2, max_irrep=np.inf):
    # max_irrep cannot be set to None since irr3 loop depends on it
    out = in1 * in2
    out = out.truncated(max_irrep)
    p = np.zeros((in1.dim, in2.dim, out.dim))
    shift3 = np.zeros(out.irreps[-1] + 1, dtype=int)
    n = 0
    for i, irr3 in enumerate(out.irreps):
        shift3[irr3] = n  # indexed with IRREP, not index
        n += out.degen[i] * irr3
    cs1 = [0, *(in1.degen * in1.irreps).cumsum()]  # remember where to restart in in1
    cs2 = [0, *(in2.degen * in2.irreps).cumsum()]  # remember where to restart in in2
    for i1, irr1 in enumerate(in1.irreps):
        for i2, irr2 in enumerate(in2.irreps):
            d2 = in2.degen[i2]
            ar = np.arange(d2)
            sl2 = slice(cs2[i2], cs2[i2] + d2 * irr2)
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_irrep + 1), 2):
                sh = (irr1, d2, irr2, d2, irr3)
                p123 = SU2_Representation.elementary_projectors[irr1, irr2, irr3]
                shift1 = cs1[i1]
                for d1 in range(in1.degen[i1]):
                    p[
                        shift1 : shift1 + irr1,
                        sl2,
                        shift3[irr3] : shift3[irr3] + d2 * irr3,
                    ].reshape(sh)[:, ar, :, ar] = p123
                    shift3[irr3] += d2 * irr3
                    shift1 += irr1
    return p


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
        return np.eye(rep_in[0].dim)

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
        proj = np.tensordot(proj, p, ((-1,), (0,)))
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
    proj : (M, N) ndarray
        Projector on singlet, with N the singlet space dimension and M the Sz=0 input
        sector dimension.
    indices : (M,) integer ndarray
        Indices of proj rows in terms of dense matrix labelling.
    """
    repL = functools.reduce(operator.mul, rep_left_enum)
    repR = functools.reduce(operator.mul, rep_right_enum)
    # save left and right dimensions before truncation
    dimL = repL.dim
    dimR = repR.dim
    target = sorted(set(repL.irreps).intersection(repR.irreps))
    if not target:
        raise ValueError("Representations have no common irrep")
    repL = repL.truncated(target[-1])
    repR = repR.truncated(target[-1])
    if target != list(repL.irreps) or target != list(repR.irreps):  # TODO
        raise NotImplementedError("TODO: remove irreps that will not fuse")

    projL_U1 = BlockMatrixU1.from_dense(
        get_projector_chained(*rep_left_enum).reshape(dimL, -1)[:, : repL.dim],
        combine_colors(*(r.get_Sz() for r in rep_left_enum)),
        repL.get_Sz(),
    )

    projR = get_projector_chained(*rep_right_enum).reshape(dimR, -1)[:, : repR.dim]
    szR_in = combine_colors(*(r.get_Sz() for r in rep_right_enum))
    if conj_right:  # same as conjugating input irrep, with smaller dimensions
        projR = projR @ repR.get_conjugator()
        szR_in = -szR_in  # read-only

    projR_U1 = BlockMatrixU1.from_dense(projR, szR_in, repR.get_Sz())
    del projR

    nrows = 0
    slice_in_blocks = []
    indices_in = []
    for i in range(projL_U1.nblocks):
        block_dim = projL_U1.blocks[i].shape[0] * projR_U1.blocks[-i - 1].shape[0]
        slice_in_blocks.append(slice(nrows, nrows + block_dim))
        indices_in.extend(
            (
                (projL_U1.row_indices[i] * projR_U1.shape[0])[:, None]
                + projR_U1.row_indices[-i - 1]
            ).ravel()
        )
        nrows += block_dim

    indices_in = np.array(indices_in)
    singlet_dim = repL.degen @ repR.degen
    shift_out = 0
    shiftL = np.zeros(projL_U1.nblocks, dtype=int)
    shiftR = np.zeros(projR_U1.nblocks, dtype=int)
    full_proj = np.zeros((nrows, singlet_dim))
    for i, irr in enumerate(target):
        degenL = repL.degen[i]
        degenR = repR.degen[i]
        slice_out = slice(shift_out, shift_out + degenL * degenR)
        shift_out += degenL * degenR
        for j in range(irr):
            mz = irr - 1 - 2 * j
            biL = projL_U1.get_color_index(mz)
            biR = projR_U1.nblocks - biL - 1
            matL = projL_U1.blocks[biL][:, shiftL[biL] : shiftL[biL] + degenL]
            matR = projR_U1.blocks[biR][:, shiftR[biR] : shiftR[biR] + degenR]
            if matL.size < matR.size:  # singlet proj = anti_diag(-1**i)/sqrt(irr)
                matL = (1 - (irr + mz) // 2 % 2 * 2) / np.sqrt(irr) * matL
            else:
                matR = (1 - (irr + mz) // 2 % 2 * 2) / np.sqrt(irr) * matR
            shiftL[biL] += degenL
            shiftR[biR] += degenR
            full_proj[slice_in_blocks[biL], slice_out] = (
                (matL.ravel()[:, None] * matR.ravel())
                .reshape(matL.shape + matR.shape)
                .swapaxes(1, 2)
                .reshape(-1, degenL * degenR)
            )

    # full_proj is *not* a valid BlockMatrixU1 because of unsorted indices in only block
    return full_proj, indices_in


def construct_transpose_matrix(
    representations, n_bra_leg1, n_bra_leg2, swap, contract=True
):
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
    contract: bool
        Whether to contract projectors. If False, non-contracted projectors are
        returned.
    """
    assert len(representations) == len(swap)
    sh1 = tuple(r.dim for r in representations)
    proj1, nnz1 = construct_matrix_projector(
        representations[:n_bra_leg1], representations[n_bra_leg1:]
    )

    rep_bra2 = tuple(representations[i] for i in swap[:n_bra_leg2])
    rep_ket2 = tuple(representations[i] for i in swap[n_bra_leg2:])
    proj2, nnz2 = construct_matrix_projector(rep_bra2, rep_ket2)

    # so, now we have initial shape projector and output shape projector, with only
    # Sz=0 block and swapped rows for both. We need to reorder rows to contract them.
    # proj1 has rows according to nnz1. proj2 has rows according to nnz2, which refers
    # to transposed full tensor. It is more efficient to swap only one matrix.

    # 1) reformulate nnz1 in terms of proj2 leg ordering
    sh2 = tuple(r.dim for r in rep_bra2) + tuple(r.dim for r in rep_ket2)
    strides1 = np.array((1,) + sh1[:0:-1]).cumprod()[::-1]
    strides2 = np.array((1,) + sh2[:0:-1]).cumprod()[::-1]
    nnz1 = (nnz1[:, None] // strides1 % sh1)[:, swap] @ strides2

    # 2) transposed nnz1 and nnz2 contain the same values in different order. We want
    # the permutation linking them. Better to sort first then use binary search.
    so1 = nnz1.argsort()
    perm = so1[nnz1.searchsorted(nnz2, sorter=so1)]

    # 3) perm sends nnz2 to sorted(transposed nnz1). We can first swap proj2 according
    # to perm, then re-swap it under argsort(so1) - or in one row swap proj2 according
    # to perm[argsort[so1] - to get transposed nnz1. Avoid 2nd sort by swapping perm.

    # Memory peak is here. Need to store proj1, proj1[perm] and proj2. It is possible
    # to construct perm before proj2 and gain a factor 3/2, but requires doing twice
    # construct_transpose_matrix work, arrays excepted.
    proj1 = proj1[perm]
    if contract:
        return proj2.T @ proj1
    return proj2, proj1


class SU2_Matrix(object):
    __array_priority__ = 15.0  # bypass ndarray.__mul__

    def __init__(self, blocks, block_irreps, rep_left, rep_right):
        # need block_irreps since some blocks may be zero
        assert len(blocks) == len(block_irreps)
        self._blocks = blocks
        self._block_irreps = block_irreps
        self._nblocks = len(blocks)
        self._rep_left = rep_left
        self._rep_right = rep_right

    @property
    def shape(self):
        return (self._rep_left.dim, self._rep_right.dim)

    @classmethod
    def from_raw_data(cls, data, rep_in, rep_out):
        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        k = 0
        while i1 < rep_in.n_irr and i2 < rep_out.n_irr:
            if rep_in.irreps[i1] == rep_out.irreps[i2]:
                sh = (rep_in.degen[i1], rep_out.degen[i2])
                m = data[k : k + sh[0] * sh[1]].reshape(sh) / np.sqrt(rep_in.irreps[i1])
                blocks.append(m)
                k += m.size
                block_irreps.append(rep_in.irreps[i1])
                i1 += 1
                i2 += 1
            elif rep_in.irreps[i1] < rep_out.irreps[i2]:
                i1 += 1
            else:
                i2 += 1
        assert k == data.size
        return cls(blocks, block_irreps, rep_in, rep_out)

    @classmethod
    def from_dense(cls, mat, rep_left_enum, rep_right_enum):
        prod_l = functools.reduce(operator.mul, rep_left_enum)
        prod_r = functools.reduce(operator.mul, rep_right_enum)
        proj, ind = construct_matrix_projector(
            rep_left_enum, rep_right_enum, conj_right=True
        )
        data = proj.T @ mat.ravel()[ind]
        return cls.from_raw_data(data, prod_l, prod_r)

    def to_raw_data(self):
        # some blocks may be allowed by SU(2) in current matrix form but be zero and
        # missing in block_irreps (matrix created by matrix product). Still, data has to
        # include the corresponding zeros at the accurate position.
        shared, indL, indR = np.intersect1d(  # bruteforce numpy > clever python
            self._rep_left.irreps,
            self._rep_right.irreps,
            assume_unique=True,
            return_indices=True,
        )
        data = np.zeros(self._rep_left.degen[indL] @ self._rep_right.degen[indR])
        k = 0
        for i, irr in enumerate(shared):
            j = bisect.bisect_left(self._block_irreps, irr)
            if j < self._nblocks and self._block_irreps[j] == irr:
                b = self._blocks[j]
                data[k : k + b.size] = b.ravel() * np.sqrt(irr)
                k += b.size
            else:  # missing block
                k += self._rep_left.degen[indL[i]] * self._rep_right.degen[indR[i]]
        return data

    def toarray(self, rep_left_enum=None, rep_right_enum=None):
        if rep_left_enum is None:
            rep_left_enum = (self._rep_left,)
        if rep_right_enum is None:
            rep_right_enum = (self._rep_right,)
        proj, ind = construct_matrix_projector(
            rep_left_enum, rep_right_enum, conj_right=True
        )
        ar = np.zeros(self._rep_left.dim * self._rep_right.dim)
        ar[ind] = proj @ self.to_raw_data()
        return ar.reshape(self.shape)

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
        return SU2_Matrix(blocks, self._block_irreps, self._rep_left, self._rep_right)

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

    def __neg__(self):
        blocks = [-b for b in self._blocks]
        return SU2_Matrix(blocks, self._block_irreps, self._rep_left, self._rep_right)

    @property
    def T(self):
        blocks = [b.T for b in self._blocks]
        return SU2_Matrix(blocks, self._block_irreps, self._rep_right, self._rep_left)

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
        return SU2_Matrix(blocks, block_irreps, self._rep_left, other._rep_right)

    def __add__(self, other):
        # not that rep_left and rep_right are product of input / output of rep. They may
        # correspond to different decompositions and addition would not be allowed for
        # dense tensors (different shapes) / meaningless for matrices.
        if self._rep_left != other._left or self._rep_right != other._right:
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
        return SU2_Matrix(blocks, block_irreps, self._rep_left, other._rep_right)

    def __sub__(self, other):
        return self + (-other)

    def expm(self):
        """
        Compute expm(self)
        """
        blocks = [lg.expm(b) for b in self._blocks]
        return SU2_Matrix(blocks, self._block_irreps, self._rep_left, self._rep_right)

    def svd(self, cut=None, rcutoff=1e-11):
        """
        Compute block-wise SVD of self and keep only cut largest singular values. Do not
        truncate if cut is not provided. Keep only values larger than rcutoff * max(sv).
        """
        block_u = [None] * self._nblocks
        block_s = [None] * self._nblocks
        block_v = [None] * self._nblocks
        block_max_vals = np.empty(self._nblocks)
        for bi, b in enumerate(self._blocks):
            block_u[bi], block_s[bi], block_v[bi] = lg.svd(b, full_matrices=False)
            block_max_vals[bi] = block_s[bi][0]

        cutoff = block_max_vals.max() * rcutoff  # cannot be set before 1st loop
        block_cuts = [0] * self._nblocks
        if cut is None:  # still remove values smaller than cutoff
            for bi, bs in enumerate(block_s):
                keep = (bs > cutoff).nonzero()[0]
                if keep.size:
                    block_cuts[bi] = keep[-1] + 1
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
        U = SU2_Matrix(block_u, mid_rep.irreps, self._rep_left, mid_rep)
        V = SU2_Matrix(block_v, mid_rep.irreps, mid_rep, self._rep_right)
        s = np.array(s[::-1])
        return U, s, V, mid_rep
