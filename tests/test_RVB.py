#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from symmetric_tensor.o2_symmetric_tensor import O2_SymmetricTensor
from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor
from ctmrg.ctmrg import CTMRG

########################################################################################
# Test CTMRG on RVB SU(2) wavefunction
# Especially relevant for merge_legs and update_signature
########################################################################################
tol = 5e-14


def check_rdm(ctm1, ctm2):
    # only check rdm eigenvalues are the same, rdm themselves may have different gauge
    # conventions.
    rdm1 = ctm1.compute_rdm2x1(0, 0)
    rdm2 = ctm2.compute_rdm2x1(0, 0)
    assert lg.norm(rdm1 - rdm1.T.conj()) < tol
    assert lg.norm(rdm2 - rdm2.T.conj()) < tol
    assert lg.norm(lg.eigvalsh(rdm1) - lg.eigvalsh(rdm2)) < tol


def check_ctm(ctm1, ctm2):
    check_rdm(ctm1, ctm2)
    for i in range(4):
        ctm1.iterate()
        ctm2.iterate()
        check_rdm(ctm1, ctm2)


# set parameters
d = 2
D = d + 1
rr_U1 = (np.arange(d - 1, -d, -2, dtype=np.int8), np.array([0], dtype=np.int8))
rc_U1 = (np.array([0, *rr_U1[0]], dtype=np.int8),) * 4
rr_O2 = (np.array([[1], [1]]), np.array([[1], [0]]))
rc_O2 = (np.array([[1, 1], [0, 1]]),) * 4
rr_SU2 = (np.array([[1], [d]]), np.array([[1], [1]]))
rc_SU2 = (np.array([[1, 1], [1, d]]),) * 4

# construct tensor
tRVB = np.zeros((d, 1, D, D, D, D))
for i in range(d):
    tRVB[i, 0, 0, 0, 0, i + 1] = 1.0
    tRVB[i, 0, 0, 0, i + 1, 0] = 1.0
    tRVB[i, 0, 0, i + 1, 0, 0] = 1.0
    tRVB[i, 0, i + 1, 0, 0, 0] = 1.0


# 1st test: AB//BA pattern with staggered signature
tA_U1 = U1_SymmetricTensor.from_array(tRVB, rr_U1, rc_U1, signature=[0, 0, 1, 1, 1, 1])
tB_U1 = U1_SymmetricTensor.from_array(tRVB, rr_U1, rc_U1, signature=[1, 1, 0, 0, 0, 0])
tA_O2 = O2_SymmetricTensor.from_array(tRVB, rr_O2, rc_O2, signature=[0, 0, 1, 1, 1, 1])
tB_O2 = O2_SymmetricTensor.from_array(tRVB, rr_O2, rc_O2, signature=[1, 1, 0, 0, 0, 0])
tA_SU2 = SU2_SymmetricTensor.from_array(
    tRVB, rr_SU2, rc_SU2, signature=[0, 0, 1, 1, 1, 1]
)
tB_SU2 = SU2_SymmetricTensor.from_array(
    tRVB, rr_SU2, rc_SU2, signature=[1, 1, 0, 0, 0, 0]
)
assert lg.norm(tA_O2.toarray() - tRVB) < tol
assert lg.norm(tB_O2.toarray() - tRVB) < tol
assert lg.norm(tA_SU2.toarray() - tRVB) < tol
assert lg.norm(tB_SU2.toarray() - tRVB) < tol
assert (tA_O2.toU1() - tA_U1).norm() < tol
assert (tB_O2.toU1() - tB_U1).norm() < tol
assert (tA_SU2.toU1() - tA_U1).norm() < tol
assert (tB_SU2.toU1() - tB_U1).norm() < tol

# check bilayer contraction
a0 = np.tensordot(tRVB, tRVB, ((0, 1), (0, 1)))
a_U1 = tA_U1.dagger() @ tA_U1
a_O2 = tA_O2.dagger() @ tA_O2
a_SU2 = tA_SU2.dagger() @ tA_SU2
assert lg.norm(a_U1.toarray() - a0) < tol
assert lg.norm(a_O2.toarray() - a0) < tol
assert (a_O2.toU1() - a_U1).norm() < tol
assert lg.norm(a_SU2.toarray() - a0) < tol
assert (a_SU2.toU1() - a_U1).norm() < tol

a1 = a0.transpose(0, 4, 1, 5, 2, 6, 3, 7)
a1_U1 = a_U1.permutate((0, 4, 1, 5), (2, 6, 3, 7))
a1_O2 = a_O2.permutate((0, 4, 1, 5), (2, 6, 3, 7))
a1_SU2 = a_SU2.permutate((0, 4, 1, 5), (2, 6, 3, 7))
assert lg.norm(a1_U1.toarray() - a1) < tol
assert lg.norm(a1_O2.toarray() - a1) < tol
assert (a1_O2.toU1() - a1_U1).norm() < tol
assert lg.norm(a1_SU2.toarray() - a1) < tol
assert (a1_SU2.toU1() - a1_U1).norm() < tol
del a0, a_U1, a_O2, a_SU2, a1, a1_U1, a1_O2, a1_SU2

# Due to SU(2) symmetry, singular values come in multiplets.
# Not breaking these multiplets is important for numerical precision, else reduced
# density matrix are hermitian up to poor precision
# this is done with degen_ratio.

# check ctm with AB//BA pattern
ctmU1_AB = CTMRG.from_elementary_tensors("AB\nBA", (tA_U1, tB_U1), 20, degen_ratio=0.99)
# workaround for missing O(2) merge_legs
ctmO2_AB = CTMRG.from_elementary_tensors(
    "AB\nBA", (tA_SU2, tB_SU2), 20, degen_ratio=0.99
)
ctmO2_AB.set_symmetry("O2")
check_ctm(ctmU1_AB, ctmO2_AB)

ctmU1_AB.restart_environment()
ctmSU2_AB = CTMRG.from_elementary_tensors(
    "AB\nBA", (tA_SU2, tB_SU2), 20, degen_ratio=0.99
)
check_ctm(ctmU1_AB, ctmSU2_AB)


# Now try 1-site unit cell by changing signature on 2 legs.
# Define Sz-reversal operation on virtual leg as
zD = np.eye(D)
zD[1:, 1:] = np.diag(1 - np.arange(d) % 2 * 2)[::-1]


# 1st: UR
s = np.array([0, 0, 0, 0, 1, 1])
tRVBur = np.einsum("paurdl,Uu,Rr->paURdl", tRVB, zD, zD)
tRVB_check = tA_SU2.copy()
tRVB_check.update_signature([0, 0, 1, 1, 0, 0])
assert lg.norm(tRVB_check.toarray() - tRVBur) < tol

tur_U1 = U1_SymmetricTensor.from_array(tRVBur, rr_U1, rc_U1, signature=s)
tur_SU2 = SU2_SymmetricTensor.from_array(tRVBur, rr_SU2, rc_SU2, signature=s)
assert lg.norm(tur_U1.toarray() - tRVBur) < tol
assert lg.norm(tur_SU2.toarray() - tRVBur) < tol
assert (tur_SU2 - tRVB_check).norm() < tol
assert (tur_SU2.toU1() - tur_U1).norm() < tol

ctmU1_ur = CTMRG.from_elementary_tensors("A", (tur_U1,), 20, degen_ratio=0.99)
ctmO2_ur = CTMRG.from_elementary_tensors("A", (tur_SU2,), 20, degen_ratio=0.99)
ctmO2_ur.set_symmetry("O2")
check_ctm(ctmU1_ur, ctmO2_ur)
ctmU1_ur.restart_environment()
ctmSU2_ur = CTMRG.from_elementary_tensors("A", (tur_SU2,), 20, degen_ratio=0.99)
check_ctm(ctmU1_ur, ctmSU2_ur)


# TODO: update_signature
"""
# 2nd: RD, with a loop inside signature change
s = np.array([0, 0, 1, 0, 0, 1])
tRVBrd = np.einsum("paurdl,rR,Dd->pauRDl", tRVB, zD, zD)
tRVB_check = tA_SU2.copy()
tRVB_check.update_signature([0, 0, 0, -1, 1, 0])
assert lg.norm(tRVB_check.toarray() - tRVBrd) < tol
trd_U1 = U1_SymmetricTensor.from_array(tRVBrd, rr_U1, rc_U1, signature=s)
trd_SU2 = SU2_SymmetricTensor.from_array(tRVBrd, rr_SU2, rc_SU2, signature=s)
assert (trd_SU2.toU1() - trd_U1).norm() < tol
ctmU1_rd = CTMRG.from_elementary_tensors("A", (trd_U1,), 20, degen_ratio=0.99)
ctmO2_rd = CTMRG.from_elementary_tensors("A", (trd_SU2,), 20, degen_ratio=0.99)
ctmO2_rd.set_symmetry("O2")
check_ctm(ctmU1_rd, ctmO2_rd)
ctmU1_rd.restart_environment()
ctmSU2_rd = CTMRG.from_elementary_tensors("A", (trd_SU2,), 20, degen_ratio=0.99)
check_ctm(ctmU1_rd, ctmSU2_rd)


# 3rd: DL
s = np.array([0, 0, 1, 1, 0, 0])
tRVBdl = np.einsum("paurdl,Dd,Ll->paurDL", tRVB, zD, zD)
tRVB_check = tA_SU2.copy()
tRVB_check.update_signature([0, 0, 0, 0, 1, 1])
assert lg.norm(tRVB_check.toarray() - tRVBdl) < tol
tdl_U1 = U1_SymmetricTensor.from_array(tRVBdl, rr_U1, rc_U1, signature=s)
tdl_SU2 = SU2_SymmetricTensor.from_array(tRVBdl, rr_SU2, rc_SU2, signature=s)
assert (tdl_SU2.toU1() - tdl_U1).norm() < tol
ctmU1_dl = CTMRG.from_elementary_tensors("A", (tdl_U1,), 20, degen_ratio=0.99)
ctmO2_dl = CTMRG.from_elementary_tensors("A", (tdl_SU2,), 20, degen_ratio=0.99)
ctmO2_dl.set_symmetry("O2")
check_ctm(ctmU1_dl, ctmO2_dl)
ctmU1_dl.restart_environment()
ctmSU2_dl = CTMRG.from_elementary_tensors("A", (tdl_SU2,), 20, degen_ratio=0.99)
check_ctm(ctmU1_dl, ctmSU2_dl)
"""
