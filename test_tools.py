#! /usr/bin/env python3
import numpy as np
import scipy.linalg as lg


# SU(2) generators
def construct_genSU2_s(s):
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


# spin 2 AKLT tensor on the square lattice
tAKLT2 = np.zeros((5, 2, 2, 2, 2), dtype=np.float64)
tAKLT2[0, 0, 0, 0, 0] = 1.0
tAKLT2[1, 0, 0, 0, 1] = 0.5
tAKLT2[1, 0, 0, 1, 0] = 0.5
tAKLT2[1, 0, 1, 0, 0] = 0.5
tAKLT2[1, 1, 0, 0, 0] = 0.5
tAKLT2[2, 0, 0, 1, 1] = 1 / np.sqrt(6)
tAKLT2[2, 0, 1, 0, 1] = 1 / np.sqrt(6)
tAKLT2[2, 0, 1, 1, 0] = 1 / np.sqrt(6)
tAKLT2[2, 1, 0, 0, 1] = 1 / np.sqrt(6)
tAKLT2[2, 1, 0, 1, 0] = 1 / np.sqrt(6)
tAKLT2[2, 1, 1, 0, 0] = 1 / np.sqrt(6)
tAKLT2[3, 0, 1, 1, 1] = 0.5
tAKLT2[3, 1, 0, 1, 1] = 0.5
tAKLT2[3, 1, 1, 0, 1] = 0.5
tAKLT2[3, 1, 1, 1, 0] = 0.5
tAKLT2[4, 1, 1, 1, 1] = 1.0

# construct AKLT Hamiltonian
gen5 = construct_genSU2_s(2)
cconj5 = np.rint(
    lg.expm(-1j * np.pi * gen5[1]).real
)  # SU(2) charge conjugation = pi-rotation over y
tAKLT2_conj = np.tensordot(cconj5, tAKLT2, ((1,), (0,)))
SdS55 = np.tensordot(gen5, gen5, (0, 0)).real.swapaxes(1, 2).reshape(25, 25)
SdS55_2 = SdS55 @ SdS55
SdS55b = np.tensordot(gen5, -gen5.conj(), (0, 0)).real.swapaxes(1, 2).reshape(25, 25)
SdS55b_2 = SdS55b @ SdS55b

H_AKLT = (
    1 / 28 * SdS55
    + 1 / 40 * SdS55_2
    + 1 / 180 * SdS55 @ SdS55_2
    + 1 / 2520 * SdS55_2 @ SdS55_2
)
H_AKLT_55b = (
    1 / 28 * SdS55b
    + 1 / 40 * SdS55b_2
    + 1 / 180 * SdS55b @ SdS55b_2
    + 1 / 2520 * SdS55b_2 @ SdS55b_2
)

# SU(2) RVB tensor
tRVB2 = np.zeros((2, 3, 3, 3, 3))
tRVB2[0, 0, 2, 2, 2] = 1.0
tRVB2[0, 2, 0, 2, 2] = 1.0
tRVB2[0, 2, 2, 0, 2] = 1.0
tRVB2[0, 2, 2, 2, 0] = 1.0
tRVB2[1, 1, 2, 2, 2] = 1.0
tRVB2[1, 2, 1, 2, 2] = 1.0
tRVB2[1, 2, 2, 1, 2] = 1.0
tRVB2[1, 2, 2, 2, 1] = 1.0

gen2 = construct_genSU2_s(1 / 2)
cconj2 = np.rint(
    lg.expm(-1j * np.pi * gen2[1]).real
)  # SU(2) charge conjugation = pi-rotation over y
SdS_22 = np.tensordot(gen2, gen2, (0, 0)).real.swapaxes(1, 2).reshape(4, 4)
SdS_22b = np.tensordot(gen2, -gen2.conj(), (0, 0)).real.swapaxes(1, 2).reshape(4, 4)
