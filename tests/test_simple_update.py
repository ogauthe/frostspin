#!/usr/bin/env python

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor
from symmetric_tensor.asymmetric_tensor import AsymmetricTensor
from simple_update.simple_update1x2 import SimpleUpdate1x2
from ctmrg.ctmrg import CTMRG

d = 2
tau = 0.01
beta = 1.0
params = {"degen_ratio": 0.99, "cutoff": 1e-10}


SdS_22 = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)
SdS_22b = np.array(
    [
        [-0.25, 0.0, 0.0, -0.5],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0],
        [-0.5, 0.0, 0.0, -0.25],
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


SdS_33b = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
    ]
)

hff = [None, None, SdS_22, SdS_33]
hfc = [None, None, SdS_22b, SdS_33b]
SdS = hff[d]
SdSb = hfc[d]
print(f"Test SimpleUpdate1x2 for SU(2) irrep {d}")
print("Benchmark Asymmetric, U1_Symmetric and SU2_Symmetric results")
print(f"evolve from beta = 0 to beta = {beta} with tau = {tau}, keeping 2 multiplets")

repd_As = np.array([d])
repd_U1 = np.arange(d - 1, -d, -2, dtype=np.int8)
repd_SU2 = np.array([[1], [d]])
hU1 = [
    U1_SymmetricTensor.from_array(
        SdSb.reshape(d, d, d, d), [repd_U1, -repd_U1], [-repd_U1, repd_U1]
    )
]
hSU2 = [
    SU2_SymmetricTensor.from_array(
        SdS.reshape(d, d, d, d), [repd_SU2, repd_SU2], [repd_SU2, repd_SU2]
    )
]
hAs = [
    AsymmetricTensor.from_array(
        SdSb.reshape(d, d, d, d), [repd_As, repd_As], [repd_As, repd_As]
    )
]

suU1 = SimpleUpdate1x2.from_infinite_temperature(4, tau, hU1)
suAs = SimpleUpdate1x2.from_infinite_temperature(4, tau, hAs)
suSU2 = SimpleUpdate1x2.from_infinite_temperature(4, tau, hSU2)

suAs.evolve(beta)
suU1.evolve(beta)
suSU2.evolve(beta)

print(f"simple update at beta = {beta}")
print("suAs weights:", *suAs._weights, sep="\n")
print("suU1 weights:", *suU1._weights, sep="\n")
print("suSU2 weights:", *suSU2._weights, sep="\n")

suU1.save_to_file("save_su.npz")
su2 = SimpleUpdate1x2.load_from_file("save_su.npz")


tensorsAs = suAs.get_tensors()
tensorsU1 = suU1.get_tensors()
tensorsSU2 = suSU2.get_tensors()
til = "AB\nBA"

ctmAs = CTMRG.from_elementary_tensors(til, tensorsAs, 30, **params)
ctmU1 = CTMRG.from_elementary_tensors(til, tensorsU1, 30, **params)
ctmSU2 = CTMRG.from_elementary_tensors(til, tensorsSU2, 30, **params)

ctm_iter = 10
print(f"Run CTMRG for {ctm_iter} iterations...")
for i in range(ctm_iter):
    ctmAs.iterate()
    ctmU1.iterate()
    ctmSU2.iterate()

print("done.", "#" * 79, sep="\n")
print("Asymmetric CTMRG:")
print(ctmAs)
print("\nU(1) CTMRG:")
print(ctmU1)
print("\nSU(2) CTMRG:")
print(ctmSU2)

print("", "#" * 79, sep="\n")
print("Compute rdm2x1 and check spectra")
rdmAs = ctmAs.compute_rdm2x1(0, 0)
rdmU1 = ctmU1.compute_rdm2x1(0, 0)
rdmSU2 = ctmSU2.compute_rdm2x1(0, 0)
print(lg.eigvalsh(rdmAs), f" {lg.norm(rdmAs-rdmAs.T.conj()):.0e}")
print(lg.eigvalsh(rdmU1), f" {lg.norm(rdmU1-rdmU1.T.conj()):.0e}")
print(lg.eigvalsh(rdmSU2), f" {lg.norm(rdmSU2-rdmSU2.T.conj()):.0e}")

print("Compute rdm1x2 and check spectra")
rdmAs = ctmAs.compute_rdm1x2(0, 0)
rdmU1 = ctmU1.compute_rdm1x2(0, 0)
rdmSU2 = ctmSU2.compute_rdm1x2(0, 0)
print(lg.eigvalsh(rdmAs), f" {lg.norm(rdmAs-rdmAs.T.conj()):.0e}")
print(lg.eigvalsh(rdmU1), f" {lg.norm(rdmU1-rdmU1.T.conj()):.0e}")
print(lg.eigvalsh(rdmSU2), f" {lg.norm(rdmSU2-rdmSU2.T.conj()):.0e}")

xihAs = ctmAs.compute_corr_length_h()
xivAs = ctmAs.compute_corr_length_v()
xihU1 = ctmU1.compute_corr_length_h()
xivU1 = ctmU1.compute_corr_length_v()
xihSU2 = ctmSU2.compute_corr_length_h()
xivSU2 = ctmSU2.compute_corr_length_v()

print("\nCorrelation lengths:")
print(f"xiAs: ({xihAs:.3f}, {xivAs:.3f})")
print(f"xiU1: ({xihU1:.3f}, {xivU1:.3f})")
print(f"xiSU2: ({xihSU2:.3f}, {xivSU2:.3f})")
