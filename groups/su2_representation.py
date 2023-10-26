import os
import json

import numpy as np
import numba


def compute_CG(max_irr):
    # load sympy only if recomputing Clebsch-Gordon is required.
    import sympy as sp
    from sympy.physics.quantum.cg import CG

    elementary_projectors = {}
    for irr1 in range(1, max_irr):
        s1 = sp.Rational(irr1 - 1, 2)

        # irr1 x irr2 = sum irr3
        for irr2 in range(1, max_irr):
            s2 = sp.Rational(irr2 - 1, 2)
            for irr3 in range(abs(irr2 - irr1) + 1, irr1 + irr2, 2):
                s3 = sp.Rational(irr3 - 1, 2)
                p = np.zeros((irr1, irr2, irr3), dtype=float)
                for i1 in range(irr1):
                    m1 = s1 - i1
                    for i2 in range(irr2):
                        m2 = s2 - i2
                        for i3 in range(irr3):
                            m3 = s3 - i3
                            p[i1, i2, i3] = CG(s1, m1, s2, m2, s3, m3).doit()  # exact
                elementary_projectors[irr1, irr2, irr3] = p  # cast to float
    return elementary_projectors


# json format is text format, easier to read and parse
# it could also be added to git
# however it generates heavy files (~177 MB vs 3.9 MB for npz for max_irr=22)
# better to have a lighter npz outside of git but easy to recompute.
def save_CG_json(savefile, elementary_projectors):
    max_irr = np.array(list(elementary_projectors.keys()))[:, 0].max()
    info = "This file contains Clebsch-Gordan coefficients for SU(2)."
    info = info + "Each irreducible represensation is labelled by its dimension."
    data = {"!CG_maximal_irrep": int(max_irr), "!info": info}
    cg_data = {}
    for k in elementary_projectors:
        nk = f"{k[0]},{k[1]},{k[2]}"
        cg_data[nk] = elementary_projectors[k].tolist()
    data["CG_data"] = cg_data
    with open(savefile, "w") as out:
        json.dump(data, out, indent=2, sort_keys=True)
    print(f"Elementary CG projectors saved in file {savefile}")
    return


def load_CG_json(savefile):
    with open(savefile) as fin:
        data = json.load(fin)

    cg_data = data["CG_data"]
    elementary_projectors = {}
    for key in cg_data:
        nk = tuple(map(int, key.split(",")))
        elementary_projectors[nk] = np.array(cg_data[key], dtype=float)
    return elementary_projectors


def save_CG_npz(savefile, elementary_projectors):
    max_irr = np.array(list(elementary_projectors.keys()))[:, 0].max()
    info = "This file contains Clebsch-Gordan coefficients for SU(2)."
    info = info + "Each irreducible represensation is labelled by its dimension."
    data = {"CG_maximal_irrep": max_irr, "info": info}
    for k in elementary_projectors:
        nk = f"_data_CG_{k[0]}_{k[1]}_{k[2]}"
        data[nk] = elementary_projectors[k]
    np.savez_compressed(savefile, **data)
    print(f"Elementary CG projectors saved in file {savefile}")
    return


def load_CG_npz(savefile):
    elementary_projectors = {}
    with np.load(savefile) as data:
        for key in filter(lambda k: k[:9] == "_data_CG_", data.files):
            nk = tuple(map(int, key[9:].split("_")))
            elementary_projectors[nk] = data[key]
    return elementary_projectors


def get_CG(max_irr=22, savefile=None):
    if savefile is None:
        savefile = os.path.join(os.path.dirname(__file__), "data_clebsch–gordan.npz")
    try:
        extension = savefile.split(".")[-1]
        if extension == "npz":
            elementary_projectors = load_CG_npz(savefile)
        elif extension == "json":
            elementary_projectors = load_CG_json(savefile)
        else:
            raise ValueError(f"Unknown file extension: {extension}")
    except FileNotFoundError:
        print(f"*** WARNING *** file {savefile} not found.")
        print(f"Recompute Clebsch-Gordon with max_irr = {max_irr}.")
        elementary_projectors = compute_CG(max_irr)
        print(f"Done. Save them in file {savefile}")
        if extension == "npz":
            save_CG_npz(savefile, elementary_projectors)
        else:
            save_CG_json(savefile, elementary_projectors)
    return elementary_projectors


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


@numba.njit  # numba product of 2 SU(2) representations
def product_degen(degen1, irreps1, degen2, irreps2):
    degen = np.zeros(irreps1[-1] + irreps2[-1] - 1, dtype=np.int64)
    for d1, irr1 in zip(degen1, irreps1):
        for d2, irr2 in zip(degen2, irreps2):
            for irr in range(abs(irr1 - irr2), irr1 + irr2 - 1, 2):
                degen[irr] += d1 * d2  # shit irr-1 <-- irr to start at 0
    return degen


class SU2_Representation:
    """
    SU(2) representation. Representations are immutable, hashable objects.
    """

    elementary_projectors = get_CG()

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

    @classmethod
    def from_string(cls, s):
        """
        Construct SU2_Representation from string representation. Absolutly NO check is
        made on input, which has to follow str syntax.
        """
        degen = []
        irreps = []
        for d, irr in (word.split("*") for word in s.split(" + ")):
            degen.append(int(d))
            irreps.append(int(irr))
        return cls(degen, irreps)

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
    def max_irrep(self):
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
        return " + ".join(f"{d}*{irr}" for (d, irr) in zip(self._degen, self._irreps))

    def __hash__(self):
        return hash(tuple(self._degen) + tuple(self._irreps))

    def has_integer_spin(self):
        return (self._irreps % 2).any()

    def has_half_integer_spin(self):
        return (self._irreps % 2 == 0).any()

    def get_irrep_degen(self, irr):
        ind = np.searchsorted(self._irreps, irr)
        if ind < self._n_irr and self._irreps[ind] == irr:
            return self._degen[ind]
        return 0

    def truncated(self, max_irrep):
        """
        Return a new representation truncated to max_irrep. If no truncation occurred,
        return self.
        """
        ind = np.searchsorted(self._irreps, max_irrep + 1)
        if ind < self._n_irr:
            return SU2_Representation(self._degen[:ind], self._irreps[:ind])
        return self

    def get_generators(self):
        gen = np.zeros((3, self._dim, self._dim), dtype=complex)
        k = 0
        for d, irr in zip(self._degen, self._irreps):
            irrep_gen = su2_irrep_generators((irr - 1) / 2)
            for i in range(d):
                gen[:, k : k + irr, k : k + irr] = irrep_gen
                k += irr
        return gen

    def get_conjugator(self):
        conj = np.zeros((self._dim, self._dim))
        k = 0
        for d, irr in zip(self._degen, self._irreps):
            irrep_conj = np.diag(1 - np.arange(irr) % 2 * 2)[::-1].copy()
            for i in range(d):
                conj[k : k + irr, k : k + irr] = irrep_conj
                k += irr
        return conj

    def get_Sz(self):
        cartan = np.empty(self._dim, dtype=np.int8)
        k = 0
        for d, irr in zip(self._degen, self._irreps):
            irrep_cartan = -np.arange(-irr + 1, irr, 2, dtype=np.int8)
            for i in range(d):
                cartan[k : k + irr] = irrep_cartan
                k += irr
        return cartan

    def get_multiplet_structure(self):
        mult = np.empty(self._degen.sum(), dtype=int)
        k = 0
        for d, irr in zip(self._degen, self._irreps):
            mult[k : k + d] = irr
            k += d
        return mult
