#!/usr/bin/env python3

import numpy as np
import sympy as sp
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
    def __init__(self, degen, irrep):
        """
        irrep must be sorted
        """
        degen2 = np.asanyarray(degen)
        irrep2 = np.asanyarray(irrep)
        if degen2.ndim != 1 or irrep2.ndim != 1:
            raise ValueError("degen and irrep must be 1D")
        if degen2.size != irrep2.size:
            raise ValueError("degen and irrep must have same size")
        ma = degen2.nonzero()[0]
        self._degen = np.ascontiguousarray(degen2[ma])
        self._irrep = np.ascontiguousarray(irrep2[ma])
        self._dim = self._degen @ self._irrep

    @property
    def dim(self):
        return self._dim

    @property
    def degen(self):
        return self._degen

    @property
    def irrep(self):
        return self._irrep

    @property
    def max_spin(self):
        return self._irrep[-1]

    def __eq__(self, other):
        if self._degen.size != other._degen.size:
            return False
        return (self._degen == other._degen).all() and (
            self._irrep == other._irrep
        ).all()

    def __mul__(self, other):
        prod_irreps = np.zeros(self._irrep[-1] + other._irrep[-1], dtype=int)
        for (d1, irr1) in zip(self._degen, self._irrep):
            for (d2, irr2) in zip(other._degen, other._irrep):
                prod_irreps[np.arange(abs(irr1 - irr2) + 1, irr1 + irr2, 2)] += d1 * d2
        return SU2_Representation(prod_irreps, np.arange(prod_irreps.size))

    def __add__(self, other):
        i1, i2 = 0, 0
        irrep = []
        degen = []
        while i1 < self._irrep.size and i2 < other._irrep.size:
            if self._irrep[i1] == self._irrep[i2]:
                irrep.append(self._irrep[i1])
                degen.append(self._degen[i1] + self._degen[i2])
                i1 += 1
                i2 += 1
            elif self._irrep[i1] < self._irrep[i2]:
                irrep.append(self._irrep[i1])
                degen.append(self._degen[i1])
                i1 += 1
            else:
                irrep.append(self._irrep[i2])
                degen.append(self._degen[i2])
                i2 += 1
        if i1 < self._irrep.size:
            irrep.extend(self._irrep[i1:])
            degen.extend(self._degen[i1:])
        if i2 < self._irrep.size:
            irrep.extend(self._irrep[i2:])
            degen.extend(self._degen[i2:])
        return SU2_Representation(degen, irrep)

    def __repr__(self):
        s = ""
        for (d1, irr1) in zip(self._degen, self._irrep):
            s += f" + {d1}*{irr1}"
        return s[3:]

    def __hash__(self):
        return hash(repr(self))  # quick and dirty

    def copy(self):
        return SU2_Representation(self._degen.copy(), self._irrep.copy())

    def get_irrep_degen(self, irr):
        ind = np.searchsorted(self._irrep, irr)
        if ind < self._irrep.size and self._irrep[ind] == irr:
            return self._degen[ind]
        return 0

    def truncate_max_spin(self, max_spin):
        """
        Truncate any spin strictly greater than max_spin. Returns updated dimension.
        """
        ind = np.searchsorted(self._irrep, max_spin + 1)
        if ind < len(self._irrep):
            self._degen = self._degen[:ind]
            self._irrep = self._irrep[:ind]
            self._dim = self._degen @ self._irrep
        return self._dim

    def get_generators(self):
        gen = np.zeros((3, self._dim, self._dim), dtype=complex)
        k = 0
        for (d, irr) in zip(self._degen, self._irrep):
            irrep_gen = su2_irrep_generators((irr - 1) / 2)
            for i in range(d):
                gen[:, k : k + irr, k : k + irr] = irrep_gen
                k += irr
        return gen

    def get_cartan(self):
        cartan = np.empty(self._dim, dtype=np.int8)
        k = 0
        for (d, irr) in zip(self._degen, self._irrep):
            irrep_cartan = -np.arange(-irr + 1, irr, 2, dtype=np.int8)
            for i in range(d):
                cartan[k : k + irr] = irrep_cartan
                k += irr
        return cartan


elementary_projectors = {(1, 1, 1): np.ones((1, 1, 1))}
elementary_conj = {1: np.ones((1, 1))}
for irr1 in range(2, 9):
    s1 = sp.Rational(irr1 - 1, 2)
    # singlet x irrep
    elementary_projectors[1, irr1, irr1] = np.eye(irr1)[None]
    elementary_projectors[irr1, 1, irr1] = np.eye(irr1)[:, None]

    # irr1 -> bar(irr1)
    singlet_proj = sp.zeros(irr1)
    for i1 in range(irr1):
        m1 = s1 - i1
        singlet_proj[i1, irr1 - i1 - 1] = CG(s1, m1, s1, -m1, 0, 0).doit()
    elementary_conj[irr1] = np.array(sp.sqrt(irr1) * singlet_proj, dtype=float)

    # irr1 x irr2 = sum irr3
    for irr2 in range(irr1, 9):
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
    shift3 = np.zeros(out.irrep[-1] + 1, dtype=int)
    n = 0
    for i, irr3 in enumerate(out.irrep):
        shift3[irr3] = n  # indexed with IRREP, not index
        n += out.degen[i] * irr3
    cs1 = [0, *(in1.degen * in1.irrep).cumsum()]  # remember where to restart in in1
    cs2 = [0, *(in2.degen * in2.irrep).cumsum()]  # remember where to restart in in2
    for i1, irr1 in enumerate(in1.irrep):
        for i2, irr2 in enumerate(in2.irrep):
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


def get_singlet_projector_chained(*rep_in):
    # fuse only on singlet
    n_rep = len(rep_in)
    if n_rep < 2:
        raise ValueError("Must fuse at least 2 representations")
    forwards, backwards = [[rep_in[0]], [rep_in[-1]]]
    for i in range(1, n_rep):
        forwards.append(forwards[i - 1] * rep_in[i])
        backwards.append(backwards[i - 1] * rep_in[-i - 1])
    if forwards[-1].irrep[0] != 1:
        raise ValueError("No singlet in product")

    # projection is made only on singlet. Remove irreps that wont fuse to 1.
    truncations = [1]
    forwards[-1].truncate_max_spin(1)
    for (f, b) in zip(reversed(forwards[:-1]), backwards[:-1]):
        trunc = b.max_spin
        f.truncate_max_spin(trunc)
        truncations.append(trunc)

    proj = np.eye(rep_in[0].dim)
    for (f, rep, trunc) in zip(forwards, rep_in[1:], reversed(truncations[:-1])):
        p = get_projector(f, rep, max_spin=trunc)
        proj = np.tensordot(proj, p, ((-1,), (0,)))
    return proj


def get_conjugator(rep):
    conjugator = np.zeros((rep.dim, rep.dim))
    k = 0
    for (d, irr) in zip(rep.degen, rep.irrep):
        irrep_conj = elementary_conj[irr]
        for i in range(d):
            conjugator[k : k + irr, k : k + irr] = irrep_conj
            k += irr
    return conjugator


def detect_rep(values, colors, ratio=1.001):
    """
    Detect SU(2) representation of a sorted array.

    Parameters:
    -----------
    values: (d,) array
        Array whose representation must be found. Must be sorted by decreasing order.
    colors: (d,) integer array
        Sz values of values.
    ratio: float, default to 1.001
        Consider two consecutive values as degenerate if v[i] / v[i+1] < ratio.
    """
    assert values.shape == colors.shape == (values.size,)
    cliff = np.array(
        [0, *(values[:-1] > ratio * values[1:]).nonzero()[0] + 1, values.size]
    )
    dimensions = cliff[1:] - cliff[:-1]
    degen = []
    irrep = []
    n = 0
    shift = np.empty(values.size + 1, dtype=int)  # sort by increasing dimension
    for irr in sorted(set(dimensions)):
        irrep.append(irr)
        d = (dimensions == irr).sum()
        degen.append(d)
        shift[irr] = n
        n += d * irr
    symetrized_values = np.empty(values.size)
    perm = np.empty(values.size, dtype=int)
    for (i, j) in zip(cliff, cliff[1:]):
        irr = j - i
        sort_colors = np.argsort(-colors[i:j])
        if (
            colors[i:j][sort_colors] != -np.arange(-irr + 1, irr, 2, dtype=np.int8)
        ).any():
            raise ValueError("level crossing")
        symetrized_values[i:j] = values[i:j].sum() / irr
        perm[shift[irr] : shift[irr] + irr] = i + sort_colors
        shift[irr] += irr
    return SU2_Representation(degen, irrep), perm, symetrized_values[perm]
