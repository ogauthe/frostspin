#!/usr/bin/env python

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor
from symmetric_tensor.asymmetric_tensor import AsymmetricTensor
from simple_update.simple_update1x2 import SimpleUpdate1x2
from ctmrg.ctmrg import CTMRG

d = 2
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

repd_As = np.array(d)
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

tau = 0.01
suU1 = SimpleUpdate1x2.from_infinite_temperature(4, tau, hU1)
suAs = SimpleUpdate1x2.from_infinite_temperature(4, tau, hAs)
suSU2 = SimpleUpdate1x2.from_infinite_temperature(2, tau, hSU2)

suAs.evolve(1.0)
suU1.evolve(1.0)
suSU2.evolve(1.0)

print("suAs weights:", *suAs._weights, sep="\n")
print("suU1 weights:", *suU1._weights, sep="\n")
print("suSU2 weights:", *suSU2._weights, sep="\n")

tensorsAs = suAs.get_tensors()
tensorsU1 = suU1.get_tensors()
tensorsSU2 = suSU2.get_tensors()
til = "AB\nBA"

ctmAs = CTMRG.from_elementary_tensors(til, tensorsAs, 27, degen_ratio=0.99)
ctmU1 = CTMRG.from_elementary_tensors(til, tensorsU1, 27, degen_ratio=0.99)
ctmSU2 = CTMRG.from_elementary_tensors(til, tensorsSU2, 15, degen_ratio=0.99)

for i in range(10):
    ctmAs.iterate()
    ctmU1.iterate()
    ctmSU2.iterate()

rdmAs = ctmAs.compute_rdm2x1(0, 0)
rdmU1 = ctmU1.compute_rdm2x1(0, 0)
rdmSU2 = ctmSU2.compute_rdm2x1(0, 0)
print(lg.eigvalsh(rdmAs), f" {lg.norm(rdmAs-rdmAs.T.conj()):.0e}")
print(lg.eigvalsh(rdmU1), f" {lg.norm(rdmU1-rdmU1.T.conj()):.0e}")
print(lg.eigvalsh(rdmSU2), f" {lg.norm(rdmSU2-rdmSU2.T.conj()):.0e}")
