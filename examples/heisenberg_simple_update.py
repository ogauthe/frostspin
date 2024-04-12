#!/usr/bin/env python

"""
Run simple update for the Heisenberg model on the square lattice
"""

import numpy as np
import scipy.linalg as lg

from froSTspin.ctmrg.ctmrg import CTMRG
from froSTspin.simple_update.simple_update import SimpleUpdate
from froSTspin.symmetric_tensor.asymmetric_tensor import AsymmetricTensor
from froSTspin.symmetric_tensor.o2_symmetric_tensor import O2_SymmetricTensor
from froSTspin.symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor
from froSTspin.symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor

d = 2  # d=2 for spin 1/2, change to d=3 for spin 1
tau = 0.01  # imaginary time step
beta = 1.0  # inverse temperature
params = {"degen_ratio": 0.99, "cutoff": 1e-10}


SdS_22 = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)

SdS_33 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

print(f"Test 2-site SimpleUpdate for SU(2) irrep {d}")
print("Benchmark Asymmetric, U1_Symmetric, O2_Symmetric and SU2_Symmetric results")
print(f"evolve from beta = 0 to beta = {beta} with tau = {tau}, keeping 2 multiplets")

if d == 2:
    SdS = SdS_22.reshape(d, d, d, d)
    reps_O2 = (np.array([[1], [1]]),) * 2
    hO2 = O2_SymmetricTensor.from_array(SdS, reps_O2, reps_O2)
elif d == 3:
    SdS = SdS_33.reshape(d, d, d, d)
    reps_O2 = (np.array([[1, 1], [-1, 2]]),) * 2

    # set O(2) conventions
    p = np.array([1, 0, 2])
    hO2 = SdS[p[:, None, None, None], p[:, None, None], p[:, None], p]
    hO2 = O2_SymmetricTensor.from_array(hO2, reps_O2, reps_O2)
else:
    raise ValueError("unknown d")

reps_As = (np.array([d]),) * 2
reps_U1 = (np.arange(d - 1, -d, -2, dtype=np.int8),) * 2
reps_SU2 = (np.array([[1], [d]]),) * 2
hAs = AsymmetricTensor.from_array(SdS, reps_As, reps_As)
hU1 = U1_SymmetricTensor.from_array(SdS, reps_U1, reps_U1)
hSU2 = SU2_SymmetricTensor.from_array(SdS, reps_SU2, reps_SU2)

D = 4
suAs = SimpleUpdate.square_lattice_first_neighbor(hAs, D, tau)
suU1 = SimpleUpdate.square_lattice_first_neighbor(hU1, D, tau)
suO2 = SimpleUpdate.square_lattice_first_neighbor(hO2, D, tau)
suSU2 = SimpleUpdate.square_lattice_first_neighbor(hSU2, D, tau)

suAs.evolve(beta)
suU1.evolve(beta)
suO2.evolve(beta)
suSU2.evolve(beta)

print(f"simple update at beta = {beta}")
print("Asymmetric bond representations =")
print(*suAs.get_bond_representations(), sep="\n")
print("U(1) bond representations =")
print(*suU1.get_bond_representations(), sep="\n")
print("O(2) bond representations =")
print(*suO2.get_bond_representations(), sep="\n", end="\n\n")
print("SU(2) bond representations =")
print(*suSU2.get_bond_representations(), sep="\n", end="\n\n")

print("suAs weights:", *suAs.get_weights(), sep="\n")
print("suU1 weights:", *suU1.get_weights(), sep="\n")
print("suO2 weights:", *suO2.get_weights(), sep="\n")
print("suSU2 weights:", *suSU2.get_weights(), sep="\n")

suU1.save_to_file("save_su.npz")
su2 = SimpleUpdate.load_from_file("save_su.npz")


til = "AB\nBA"
chi = 30
ctmAs = CTMRG.from_elementary_tensors(til, suAs.get_tensors(), chi, **params)
ctmU1 = CTMRG.from_elementary_tensors(til, suU1.get_tensors(), chi, **params)
ctmO2 = CTMRG.from_elementary_tensors(til, suO2.get_tensors(), chi, **params)
tensorsSU2 = suSU2.get_tensors()
ctmU1_fromSU2 = CTMRG.from_elementary_tensors(til, tensorsSU2, chi, **params)
ctmU1_fromSU2.set_symmetry("U1")
ctmO2_fromSU2 = CTMRG.from_elementary_tensors(til, tensorsSU2, chi, **params)
ctmO2_fromSU2.set_symmetry("O2")
ctmSU2 = CTMRG.from_elementary_tensors(til, tensorsSU2, chi, **params)


def get_tol(rdm):
    return f"{lg.norm(rdm-rdm.T.conj()):.0e}"


rdmAs = ctmAs.compute_rdm1x2(0, 0)
rdmU1 = ctmU1.compute_rdm1x2(0, 0)
rdmO2 = ctmO2.compute_rdm1x2(0, 0)
rdmU1_2 = ctmU1_fromSU2.compute_rdm1x2(0, 0)
rdmO2_2 = ctmO2_fromSU2.compute_rdm1x2(0, 0)
rdmSU2 = ctmSU2.compute_rdm1x2(0, 0)

print("Initial reduced density matrix spectrum:")
print("Asym...........", lg.eigvalsh(rdmAs), get_tol(rdmAs))
print("U(1)...........", lg.eigvalsh(rdmU1), get_tol(rdmU1))
print("O(2)...........", lg.eigvalsh(rdmO2), get_tol(rdmO2))
print("U(1) from SU(2)", lg.eigvalsh(rdmU1_2), get_tol(rdmU1_2))
print("O(2) from SU(2)", lg.eigvalsh(rdmO2_2), get_tol(rdmO2_2))
print("SU(2)..........", lg.eigvalsh(rdmSU2), get_tol(rdmSU2))


ctm_iter = 10
print(f"\nRun CTMRG for {ctm_iter} iterations...")
for i in range(ctm_iter):
    ctmAs.iterate()
    ctmU1.iterate()
    ctmO2.iterate()
    ctmU1_fromSU2.iterate()
    ctmO2_fromSU2.iterate()
    ctmSU2.iterate()

print("done.", "#" * 79, sep="\n")
print("Asymmetric CTMRG:")
print(ctmAs)
reps = ctmAs.get_corner_representations()
print("corner representations:", *reps, sep="\n")

print("\nU(1) CTMRG:")
print(ctmU1)
reps = ctmU1.get_corner_representations()
print("corner representations:", *reps, sep="\n")

print("\nO(2) CTMRG:")
print(ctmO2)
reps = ctmO2.get_corner_representations()
print("representations", *reps, sep="\n")

print("\nSU(2) SU + U(1) CTMRG:")
print(ctmU1_fromSU2)
reps = ctmU1_fromSU2.get_corner_representations()
print("corner representations:", *reps, sep="\n")

print("\nSU(2) SU + O(2) CTMRG:")
print(ctmO2_fromSU2)
reps = ctmO2_fromSU2.get_corner_representations()
print("corner representations:", *reps, sep="\n")

print("\nSU(2) CTMRG:")
print(ctmSU2)
reps = ctmSU2.get_corner_representations()
print("corner representations:", *reps, sep="\n")

print("", "#" * 79, sep="\n")
print("Compute rdm2x1 and check spectra")
rdmAs = ctmAs.compute_rdm1x2(0, 0)
rdmU1 = ctmU1.compute_rdm1x2(0, 0)
rdmO2 = ctmO2.compute_rdm1x2(0, 0)
rdmU1_2 = ctmU1_fromSU2.compute_rdm1x2(0, 0)
rdmO2_2 = ctmO2_fromSU2.compute_rdm1x2(0, 0)
rdmSU2 = ctmSU2.compute_rdm1x2(0, 0)
print("Asym...........", lg.eigvalsh(rdmAs), get_tol(rdmAs))
print("U(1)...........", lg.eigvalsh(rdmU1), get_tol(rdmU1))
print("O(2)...........", lg.eigvalsh(rdmO2), get_tol(rdmO2))
print("U(1) from SU(2)", lg.eigvalsh(rdmU1_2), get_tol(rdmU1_2))
print("O(2) from SU(2)", lg.eigvalsh(rdmO2_2), get_tol(rdmO2_2))
print("SU(2)..........", lg.eigvalsh(rdmSU2), get_tol(rdmSU2))

print("Compute rdm1x2 and check spectra")
rdmAs = ctmAs.compute_rdm2x1(0, 0)
rdmU1 = ctmU1.compute_rdm2x1(0, 0)
rdmO2 = ctmO2.compute_rdm2x1(0, 0)
rdmU1_2 = ctmU1_fromSU2.compute_rdm2x1(0, 0)
rdmO2_2 = ctmO2_fromSU2.compute_rdm2x1(0, 0)
rdmSU2 = ctmSU2.compute_rdm2x1(0, 0)
print("Asym...........", lg.eigvalsh(rdmAs), get_tol(rdmAs))
print("U(1)...........", lg.eigvalsh(rdmU1), get_tol(rdmU1))
print("O(2)...........", lg.eigvalsh(rdmO2), get_tol(rdmO2))
print("U(1) from SU(2)", lg.eigvalsh(rdmU1_2), get_tol(rdmU1_2))
print("O(2) from SU(2)", lg.eigvalsh(rdmO2_2), get_tol(rdmO2_2))
print("SU(2)..........", lg.eigvalsh(rdmSU2), get_tol(rdmSU2))

xihAs = ctmAs.compute_corr_length_h(0)
xivAs = ctmAs.compute_corr_length_v(0)
xihU1 = ctmU1.compute_corr_length_h(0)
xivU1 = ctmU1.compute_corr_length_v(0)
xihO2 = ctmO2.compute_corr_length_h(0)
xivO2 = ctmO2.compute_corr_length_v(0)
xihU1_2 = ctmU1_fromSU2.compute_corr_length_h(0)
xivU1_2 = ctmU1_fromSU2.compute_corr_length_v(0)
xihO2_2 = ctmO2_fromSU2.compute_corr_length_h(0)
xivO2_2 = ctmO2_fromSU2.compute_corr_length_v(0)
xihSU2 = ctmSU2.compute_corr_length_h(0)
xivSU2 = ctmSU2.compute_corr_length_v(0)

print("\nCorrelation lengths:")
print(f"Asym........... ({xihAs:.3f}, {xivAs:.3f})")
print(f"U(1)........... ({xihU1:.3f}, {xivU1:.3f})")
print(f"O(2)........... ({xihO2:.3f}, {xivO2:.3f})")
print(f"U(1) from SU(2) ({xihU1_2:.3f}, {xivU1_2:.3f})")
print(f"O(2) from SU(2) ({xihO2_2:.3f}, {xivO2_2:.3f})")
print(f"SU(2).......... ({xihSU2:.3f}, {xivSU2:.3f})")
