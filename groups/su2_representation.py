import os
import bisect
import operator
import functools

import numpy as np
import scipy.linalg as lg
from numba import jit

from groups.toolsU1 import combine_colors


def compute_CG(max_irr=22):
    # load sympy only if recomputing Clebsch-Gordon is required.
    import sympy as sp
    from sympy.physics.quantum.cg import CG

    elementary_projectors = {(1, 1, 1): np.ones((1, 1, 1))}
    elementary_conj = {1: np.ones((1, 1))}
    for irr1 in range(2, max_irr):
        s1 = sp.Rational(irr1 - 1, 2)
        # singlet x irrep
        elementary_projectors[1, irr1, irr1] = np.eye(irr1)[None]
        elementary_projectors[irr1, 1, irr1] = np.eye(irr1)[:, None]

        # irr1 -> bar(irr1)
        singlet_proj = sp.zeros(irr1)
        for i1 in range(irr1):
            m1 = s1 - i1
            singlet_proj[i1, irr1 - i1 - 1] = CG(s1, m1, s1, -m1, 0, 0).doit()
        elementary_conj[irr1] = np.array(sp.sqrt(irr1) * singlet_proj.T, dtype=float)

        # irr1 x irr2 = sum irr3
        for irr2 in range(irr1, max_irr):
            s2 = sp.Rational(irr2 - 1, 2)
            for irr3 in range(irr2 - irr1 + 1, irr1 + irr2, 2):
                s3 = sp.Rational(irr3 - 1, 2)
                p = np.zeros((irr1, irr2, irr3))
                for i1 in range(irr1):
                    m1 = s1 - i1
                    for i2 in range(irr2):
                        m2 = s2 - i2
                        for i3 in range(irr3):
                            m3 = s3 - i3
                            p[i1, i2, i3] = CG(s1, m1, s2, m2, s3, m3).doit()
                elementary_projectors[irr1, irr2, irr3] = p
                elementary_projectors[irr2, irr1, irr3] = p.swapaxes(0, 1).copy()
    return elementary_projectors, elementary_conj


def save_CG(savefile, elementary_projectors, elementary_conj):
    max_irr = np.array(list(elementary_projectors.keys()))[:, 0].max()
    data = {"_CG_max_irr": max_irr}
    for k in elementary_projectors:
        nk = f"_CG_proj_{k[0]}_{k[1]}_{k[2]}"
        data[nk] = elementary_projectors[k]
    for k in elementary_conj:
        nk = f"_CG_conj_{k}"
        data[nk] = elementary_conj[k]
    np.savez_compressed(savefile, max_irr=max_irr, **data)
    print(f"saved elementary_projectors and elementary_conj in file {savefile}")


def load_CG(savefile):
    elementary_projectors, elementary_conj = {}, {}
    with np.load(savefile) as data:
        for key in filter(lambda k: k[:9] in ("_CG_proj_", "_CG_conj_"), data.files):
            indices = tuple(map(int, key[9:].split("_")))
            if len(indices) == 3:
                elementary_projectors[indices] = data[key]
            elif len(indices) == 1:
                elementary_conj[indices[0]] = data[key]
            else:
                raise KeyError
    return elementary_projectors, elementary_conj


def get_CG(max_irr=22, savefile=None):
    if savefile is None:
        savefile = os.path.join(os.path.dirname(__file__), "_data_CG.npz")
    try:
        elementary_projectors, elementary_conj = load_CG(savefile)
    except FileNotFoundError:
        print(f"File {savefile} not found.")
        print(f"Recompute Clebsch-Gordon with max_irr = {max_irr}.")
        elementary_projectors, elementary_conj = compute_CG(max_irr)
        print(f"Done. Save them in file {savefile}")
        save_CG(savefile, elementary_projectors, elementary_conj)
    return elementary_projectors, elementary_conj


def su2_irrep_generators(s):
    """
    Construct generator for spin-s irrep of SU(2)
    """
    if s < 0 or int(2 * s) != 2 * s:
        raise ValueError("s must be a positive half integer")

    d = int(2 * s) + 1
    basis = np.arange(s, -s - 1, -1)
    Sm = np.zeros((d, d))
    Sm[np.arange(1, d), np.arange(d - 1)] = np.sqrt(
        s * (s + 1) - basis[:-1] * (basis[:-1] - 1)
    )
    Sp = Sm.T
    gen = np.empty((3, d, d), dtype=complex)
    gen[0] = (Sp + Sm) / 2  # Sx
    gen[1] = (Sp - Sm) / 2j  # Sy
    gen[2] = np.diag(basis)  # Sz
    return gen


@jit(nopython=True)  # numba product of 2 SU(2) representations
def product_degen(degen1, irreps1, degen2, irreps2):
    degen = np.zeros(irreps1[-1] + irreps2[-1] - 1, dtype=np.int64)
    for (d1, irr1) in zip(degen1, irreps1):
        for (d2, irr2) in zip(degen2, irreps2):
            for irr in range(abs(irr1 - irr2), irr1 + irr2 - 1, 2):
                degen[irr] += d1 * d2  # shit irr-1 <-- irr to start at 0
    return degen


class SU2_Representation(object):
    elementary_projectors, elementary_conj = get_CG()

    def __init__(self, degen, irreps):
        """
        irrep must be sorted
        """
        degen2 = np.asanyarray(degen)
        irreps2 = np.asanyarray(irreps)
        if degen2.ndim != 1 or irreps2.ndim != 1:
            raise ValueError("degen and irreps must be 1D")
        if degen2.size != irreps2.size:
            raise ValueError("degen and irreps must have same size")
        ma = degen2.nonzero()[0]
        self._degen = np.ascontiguousarray(degen2[ma])
        self._irreps = np.ascontiguousarray(irreps2[ma])
        self._dim = self._degen @ self._irreps
        self._n_irr = self._irreps.size

    @classmethod
    def irrep(cls, irr):
        return cls([1], [irr])

    @property
    def dim(self):
        return self._dim

    @property
    def n_irr(self):
        return self._n_irr

    @property
    def degen(self):
        return self._degen

    @property
    def irreps(self):
        return self._irreps

    @property
    def max_spin(self):
        return self._irreps[-1]

    def __eq__(self, other):
        if self._n_irr != other._n_irr:
            return False
        return (self._degen == other._degen).all() and (
            self._irreps == other._irreps
        ).all()

    def __mul__(self, other):  # numba wrapper
        degen = product_degen(self._degen, self._irreps, other._degen, other._irreps)
        irreps = np.arange(1, degen.size + 1)
        return SU2_Representation(degen, irreps)

    def __add__(self, other):
        i1, i2 = 0, 0
        irreps = []
        degen = []
        while i1 < self._n_irr and i2 < other._n_irr:
            if self._irreps[i1] == other._irreps[i2]:
                irreps.append(self._irreps[i1])
                degen.append(self._degen[i1] + other._degen[i2])
                i1 += 1
                i2 += 1
            elif self._irreps[i1] < other._irreps[i2]:
                irreps.append(self._irreps[i1])
                degen.append(self._degen[i1])
                i1 += 1
            else:
                irreps.append(other._irreps[i2])
                degen.append(other._degen[i2])
                i2 += 1
        if i1 < self._n_irr:
            irreps.extend(self._irreps[i1:])
            degen.extend(self._degen[i1:])
        if i2 < other._n_irr:
            irreps.extend(other._irreps[i2:])
            degen.extend(other._degen[i2:])
        return SU2_Representation(degen, irreps)

    def __repr__(self):
        s = ""
        for (d1, irr1) in zip(self._degen, self._irreps):
            s += f" + {d1}*{irr1}"
        return s[3:]

    def __hash__(self):
        return hash(repr(self))  # quick and dirty

    def copy(self):  # to save copy before truncation
        return SU2_Representation(self._degen.copy(), self._irreps.copy())

    def get_irrep_degen(self, irr):
        ind = np.searchsorted(self._irreps, irr)
        if ind < self._n_irr and self._irreps[ind] == irr:
            return self._degen[ind]
        return 0

    def truncate_max_spin(self, max_spin):
        """
        Truncate any spin strictly greater than max_spin. Returns updated dimension.
        """
        ind = np.searchsorted(self._irreps, max_spin + 1)
        if ind < self._n_irr:
            self._degen = self._degen[:ind]
            self._irreps = self._irreps[:ind]
            self._dim = self._degen @ self._irreps
        return self._dim

    def get_generators(self):
        gen = np.zeros((3, self._dim, self._dim), dtype=complex)
        k = 0
        for (d, irr) in zip(self._degen, self._irreps):
            irrep_gen = su2_irrep_generators((irr - 1) / 2)
            for i in range(d):
                gen[:, k : k + irr, k : k + irr] = irrep_gen
                k += irr
        return gen

    def get_conjugator(self):
        conj = np.zeros((self._dim, self._dim))
        k = 0
        for (d, irr) in zip(self._degen, self._irreps):
            irrep_conj = SU2_Representation.elementary_conj[irr]
            for i in range(d):
                conj[k : k + irr, k : k + irr] = irrep_conj
                k += irr
        return conj

    def get_Sz(self):
        cartan = np.empty(self._dim, dtype=np.int8)
        k = 0
        for (d, irr) in zip(self._degen, self._irreps):
            irrep_cartan = -np.arange(-irr + 1, irr, 2, dtype=np.int8)
            for i in range(d):
                cartan[k : k + irr] = irrep_cartan
                k += irr
        return cartan

    def get_multiplet_structure(self):
        mult = np.empty(self._degen.sum(), dtype=int)
        k = 0
        for (d, irr) in zip(self._degen, self._irreps):
            mult[k : k + d] = irr
            k += d
        return mult


def get_projector(in1, in2, max_spin=np.inf):
    # max_spin cannot be set to None since irr3 loop depends on it
    out = in1 * in2
    out.truncate_max_spin(max_spin)
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
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_spin + 1), 2):
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
    forwards, backwards = [[rep_in[0].copy()], [rep_in[-1]]]
    for i in range(1, len(rep_in)):
        forwards.append(forwards[i - 1] * rep_in[i])
        backwards.append(backwards[i - 1] * rep_in[-i - 1])

    if singlet_only:
        # projection is made only on singlet. Remove irreps that wont fuse to 1.
        if forwards[-1].irreps[0] != 1:
            raise ValueError("No singlet in product")
        truncations = [1]
        forwards[-1].truncate_max_spin(1)
        for (f, b) in zip(reversed(forwards[:-1]), backwards[:-1]):
            trunc = b.max_spin
            f.truncate_max_spin(trunc)
            truncations.append(trunc)
    else:
        truncations = [np.inf] * len(rep_in)

    proj = np.eye(rep_in[0].dim)
    for (f, rep, trunc) in zip(forwards, rep_in[1:], reversed(truncations[:-1])):
        p = get_projector(f, rep, max_spin=trunc)
        proj = np.tensordot(proj, p, ((-1,), (0,)))
    return proj


def construct_matrix_projector(rep_left_enum, rep_right_enum, conj_right=False):
    r"""
                list of matrices
                /          \
               /            \
            prod_l        prod_r
             /               /
            /\              /\
           /\ \            /\ \
         rep_left        rep_right
    """
    prod_l = rep_left_enum[0].copy()  # need copy to truncate if n_rep_left = 1
    for rep in rep_left_enum[1:]:
        prod_l = prod_l * rep
    prod_r = rep_right_enum[0].copy()
    for rep in rep_right_enum[1:]:
        prod_r = prod_r * rep
    # save left and right dimensions before truncation
    ldim = prod_l.dim
    rdim = prod_r.dim
    target = sorted(set(prod_l.irreps).intersection(prod_r.irreps))
    if not target:
        raise ValueError("Representations have no common irrep")
    # optimal would be to fuse only on target. Currently only truncate to max_spin
    prod_l.truncate_max_spin(target[-1])
    prod_r.truncate_max_spin(target[-1])
    proj_l = get_projector_chained(*rep_left_enum)[..., : prod_l.dim]
    proj_r = get_projector_chained(*rep_right_enum)[..., : prod_r.dim]
    proj = get_projector(prod_l, prod_r, max_spin=1)
    singlet_dim = proj.shape[2]

    # contract projectors following optimal contraction path
    in_sh = proj_l.shape[:-1] + proj_r.shape[:-1]
    proj_l = proj_l.reshape(ldim, prod_l.dim)
    proj_r = proj_r.reshape(rdim, prod_r.dim)
    if conj_right:  # same as conjugating input irrep, with smaller dimensions
        proj_r = proj_r @ prod_r.get_conjugator()
    cost_lr = prod_r.dim * ldim * (prod_l.dim + rdim)
    cost_rl = prod_l.dim * rdim * (prod_r.dim + ldim)
    if cost_lr < cost_rl:
        proj = proj_l @ proj.reshape(prod_l.dim, prod_r.dim * singlet_dim)
        # del proj_l
        proj = proj.reshape(ldim, prod_r.dim, singlet_dim).swapaxes(0, 1).copy()
        proj = proj_r @ proj.reshape(prod_r.dim, ldim * singlet_dim)
        proj = proj.reshape(rdim, ldim, singlet_dim).swapaxes(0, 1).copy()
    else:
        proj = proj.swapaxes(0, 1).reshape(prod_r.dim, prod_l.dim * singlet_dim)
        proj = (proj_r @ proj).reshape(rdim, prod_l.dim, singlet_dim)
        # del proj_r
        proj = proj.swapaxes(0, 1).reshape(prod_l.dim, rdim * singlet_dim)
        proj = proj_l @ proj
    proj = proj.reshape(*in_sh, singlet_dim)
    return proj


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
    proj1 = construct_matrix_projector(
        representations[:n_bra_leg1], representations[n_bra_leg1:]
    )

    # reduce to Sz=0 rows, others are empty
    nnz_indices = (
        combine_colors(*(rep.get_Sz() for rep in representations)) == 0
    ).nonzero()[0]
    sh1 = proj1.shape[:-1]
    proj1 = proj1.reshape(-1, proj1.shape[-1])[nnz_indices]

    proj2 = construct_matrix_projector(
        [representations[i] for i in swap[:n_bra_leg2]],
        [representations[i] for i in swap[n_bra_leg2:]],
    )
    # reduce proj2 to Sz=0 block AND sparse transpose it
    cumprod1 = np.array((1,) + sh1[:0:-1]).cumprod()[::-1]
    cumprod2 = np.array((1,) + proj2.shape[-2:0:-1]).cumprod()[::-1]
    nnz_indices = (nnz_indices[:, None] // cumprod1 % sh1)[:, swap] @ cumprod2
    proj2 = proj2.reshape(-1, proj2.shape[-1])[nnz_indices]
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
        p = construct_matrix_projector(rep_left_enum, rep_right_enum, conj_right=True)
        p = p.reshape(-1, p.shape[-1])
        data = p.T @ mat.ravel()
        return cls.from_raw_data(data, prod_l, prod_r)

    def to_raw_data(self):
        # some blocks may be allowed by SU(2) in current matrix form but be zero and
        # missing in block_irreps (matrix created by matrix product). Still, data has to
        # include the corresponding zeros at the accurate position.
        data = []
        i1, i2 = 0, 0
        while i1 < self._rep_left.n_irr and i2 < self._rep_right.n_irr:
            if self._rep_left.irreps[i1] == self._rep_right.irreps[i2]:
                j = bisect.bisect_left(self._block_irreps, self._rep_left.irreps[i1])
                if (
                    j < self._nblocks
                    and self._block_irreps[j] == self._rep_left.irreps[i1]
                ):
                    b = self._blocks[j]
                    data.extend(b.ravel() * np.sqrt(self._block_irreps[j]))
                else:  # missing block
                    data.extend(
                        [0.0] * (self._rep_left.degen[i1] * self._rep_right.degen[i2])
                    )
                i1 += 1
                i2 += 1
            elif self._rep_left.irreps[i1] < self._rep_right.irreps[i2]:
                i1 += 1
            else:
                i2 += 1
        return np.array(data)

    def toarray(self, rep_left_enum=None, rep_right_enum=None):
        if rep_left_enum is None:
            rep_left_enum = (self._rep_left,)
        if rep_right_enum is None:
            rep_right_enum = (self._rep_right,)
        p = construct_matrix_projector(rep_left_enum, rep_right_enum, True)
        t = np.dot(p, self.to_raw_data())
        return t.reshape(self.shape)

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

    def norm2(self):
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
