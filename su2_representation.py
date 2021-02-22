#!/usr/bin/env python3

import numpy as np
import sympy as sp
import scipy.linalg as lg
from sympy.physics.quantum.cg import CG


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


class SU2_Representation(object):
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

    def __mul__(self, other):
        prod_irreps = np.zeros(self._irreps[-1] + other._irreps[-1], dtype=int)
        for (d1, irr1) in zip(self._degen, self._irreps):
            for (d2, irr2) in zip(other._degen, other._irreps):
                prod_irreps[np.arange(abs(irr1 - irr2) + 1, irr1 + irr2, 2)] += d1 * d2
        return SU2_Representation(prod_irreps, np.arange(prod_irreps.size))

    def __add__(self, other):
        i1, i2 = 0, 0
        irreps = []
        degen = []
        while i1 < self._n_irr and i2 < other._n_irr:
            if self._irreps[i1] == self._irreps[i2]:
                irreps.append(self._irreps[i1])
                degen.append(self._degen[i1] + other._degen[i2])
                i1 += 1
                i2 += 1
            elif self._irrep[i1] < self._irreps[i2]:
                irreps.append(self._irreps[i1])
                degen.append(self._degen[i1])
                i1 += 1
            else:
                irreps.append(self._irreps[i2])
                degen.append(self._degen[i2])
                i2 += 1
        if i1 < self._n_irr:
            irreps.extend(self._irreps[i1:])
            degen.extend(self._degen[i1:])
        if i2 < self._n_irr:
            irreps.extend(self._irreps[i2:])
            degen.extend(self._degen[i2:])
        return SU2_Representation(degen, irreps)

    def __repr__(self):
        s = ""
        for (d1, irr1) in zip(self._degen, self._irreps):
            s += f" + {d1}*{irr1}"
        return s[3:]

    def __hash__(self):
        return hash(repr(self))  # quick and dirty

    def copy(self):
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
            irrep_conj = elementary_conj[irr]
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

    def get_multiplets_structure(self):
        mult = np.empty(self._degen.sum(), dtype=int)
        k = 0
        for (d, irr) in zip(self._degen, self._irreps):
            mult[k : k + d] = irr
            k += d
        return mult


elementary_projectors = {(1, 1, 1): np.ones((1, 1, 1))}
elementary_conj = {1: np.ones((1, 1))}
ms = 10
for irr1 in range(2, ms):
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
    for irr2 in range(irr1, ms):
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
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_spin + 1), 2):
                p123 = elementary_projectors[irr1, irr2, irr3]
                shift1 = cs1[i1]
                for d1 in range(in1.degen[i1]):
                    shift2 = cs2[i2]
                    for d2 in range(in2.degen[i2]):
                        p[
                            shift1 : shift1 + irr1,
                            shift2 : shift2 + irr2,
                            shift3[irr3] : shift3[irr3] + irr3,
                        ] = p123
                        shift3[irr3] += irr3
                        shift2 += irr2
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
    prod_l = rep_left_enum[0]
    for rep in rep_left_enum[1:]:
        prod_l = prod_l * rep
    prod_r = rep_right_enum[0]
    for rep in rep_right_enum[1:]:
        prod_r = prod_r * rep
    # save left and right dimensions before truncation
    ldim = prod_l.dim
    rdim = prod_r.dim
    target = sorted(set(prod_l.irreps).intersection(prod_r.irreps))
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
        proj = proj.reshape(ldim, prod_r.dim, singlet_dim).swapaxes(0, 1).copy()
        proj = proj_r @ proj.reshape(prod_r.dim, ldim * singlet_dim)
        proj = proj.reshape(rdim, ldim, singlet_dim).swapaxes(0, 1).copy()
    else:
        proj = proj.swapaxes(0, 1).reshape(prod_r.dim, prod_l.dim * singlet_dim)
        proj = (proj_r @ proj).reshape(rdim, prod_l.dim, singlet_dim)
        proj = proj.swapaxes(0, 1).reshape(prod_l.dim, rdim * singlet_dim)
        proj = proj_l @ proj
    proj = proj.reshape(*in_sh, singlet_dim)
    return proj


class SU2_Matrix(object):
    __array_priority__ = 15.0  # bypass ndarray.__mul__

    def __init__(self, blocks, block_irreps):
        assert len(blocks) == len(block_irreps)
        self._blocks = blocks
        self._block_irreps = block_irreps
        self._nblocks = len(blocks)

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
        return cls(blocks, block_irreps)

    @classmethod
    def from_dense(cls, mat, rep_left_enum, rep_right_enum):
        prod_l = rep_left_enum[0]
        d_left = [prod_l.dim]
        for rep in rep_left_enum[1:]:
            prod_l = prod_l * rep
            d_left.append(rep.dim)
        prod_r = rep_right_enum[0]
        d_right = [prod_r.dim]
        for rep in rep_right_enum[1:]:
            prod_r = prod_r * rep
            d_right.append(rep.dim)
        p = construct_matrix_projector(rep_left_enum, rep_right_enum, conj_right=True)
        sh = d_left + d_right
        data = np.tensordot(p, mat.reshape(sh), (range(len(sh)), range(len(sh))))
        return cls.from_raw_data(data, prod_l, prod_r)

    def to_raw_data(self):
        data = np.empty(sum(b.size for b in self._blocks))
        k = 0
        for irr, b in zip(self._block_irreps, self._blocks):
            data[k : k + b.size] = b.ravel() * np.sqrt(irr)
            k += b.size
        assert k == data.size
        return data

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
        return SU2_Matrix(blocks, self._block_irreps)

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

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
        return SU2_Matrix(blocks, block_irreps)

    def expm(self):
        """
        Compute expm(self)
        """
        blocks = [lg.expm(b) for b in self._blocks]
        return SU2_Matrix(blocks, self._block_irreps)

    def svd(self, Dstar=None):
        """
        Compute block-wise SVD of self and keep only D* largest singular values. Do not
        truncate if D* is not provided.
        """
        ul = [None] * self._nblocks
        sl = [None] * self._nblocks
        vl = [None] * self._nblocks
        degen = []
        for i, b in enumerate(self._blocks):
            ul[i], sl[i], vl[i] = lg.svd(b, full_matrices=False)
            degen.append(sl[i].size)
        U = SU2_Matrix(ul, self._block_irreps)
        mid_rep = SU2_Representation(degen, self._block_irreps)
        V = SU2_Matrix(ul, self._block_irreps)
        return U, sl, V, mid_rep
