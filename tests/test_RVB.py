#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor
from ctmrg.ctmrg import CTMRG

########################################################################################
# Test CTMRG on RVB SU(2) wavefunction
# Especially relevant for merge_legs and update_signature
########################################################################################

np.set_printoptions(precision=3, suppress=True)


def check_ctm(ctmU1, ctmSU2):
    rdmU1 = ctmU1.compute_rdm2x1(0, 0)
    rdmSU2 = ctmSU2.compute_rdm2x1(0, 0)
    assert lg.norm(rdmU1 - rdmU1.T.conj()) < 1e-12
    assert lg.norm(rdmSU2 - rdmSU2.T.conj()) < 1e-12
    assert lg.norm(lg.eigvalsh(rdmU1) - lg.eigvalsh(rdmSU2)) < 1e-12

    x = ctmSU2._env.get_C1(0, 0) @ ctmSU2._env.get_C4(0, 1)
    y = ctmU1._env.get_C1(0, 0) @ ctmU1._env.get_C4(0, 1)
    print("C1-C4", x.svd()[1][0], y.svd()[1][2])

    x = ctmSU2._env.get_C2(1, 0) @ ctmSU2._env.get_C1(0, 0)
    y = ctmU1._env.get_C2(1, 0) @ ctmU1._env.get_C1(0, 0)
    print("C2-C1", x.svd()[1][0], y.svd()[1][2])

    x = ctmSU2._env.get_C3(1, 1).T @ ctmSU2._env.get_C2(1, 0)
    y = ctmU1._env.get_C3(1, 1).T @ ctmU1._env.get_C2(1, 0)
    print("C3-C2", x.svd()[1][0], y.svd()[1][2])

    x = ctmSU2._env.get_C4(0, 1) @ ctmSU2._env.get_C3(1, 1).T
    y = ctmU1._env.get_C4(0, 1) @ ctmU1._env.get_C3(1, 1).T
    print("C4-C3", x.svd()[1][0], y.svd()[1][2])

    ulu1 = ctmU1.construct_reduced_ul(0, 0)
    print("U(1):", ulu1.svd()[1][4])

    ulsu2 = ctmSU2.construct_reduced_ul(0, 0)
    ursu2 = ctmSU2.construct_reduced_ur(0, 0)
    drsu2 = ctmSU2.construct_reduced_dr(0, 0)
    dlsu2 = ctmSU2.construct_reduced_dl(0, 0)

    print("ul:", ulsu2.svd()[1][0])
    print("ur:", ursu2.svd()[1][0])
    print("dl:", dlsu2.svd()[1][0])
    print("dr:", drsu2.svd()[1][0])

    print()
    for i in range(4):
        ctmU1.iterate()
        ctmSU2.iterate()
        rdmU1 = ctmU1.compute_rdm2x1(0, 0)
        rdmSU2 = ctmSU2.compute_rdm2x1(0, 0)
        print(lg.eigvalsh(rdmU1), lg.eigvalsh(rdmSU2))


# set parameters
d = 2
D = d + 1
rr_U1 = (np.arange(d - 1, -d, -2, dtype=np.int8), np.array([0], dtype=np.int8))
rc_U1 = (np.array([0, *rr_U1[0]], dtype=np.int8),) * 4
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
tA_SU2 = SU2_SymmetricTensor.from_array(
    tRVB, rr_SU2, rc_SU2, signature=[0, 0, 1, 1, 1, 1]
)
tB_SU2 = SU2_SymmetricTensor.from_array(
    tRVB, rr_SU2, rc_SU2, signature=[1, 1, 0, 0, 0, 0]
)
assert lg.norm(tA_SU2.toarray() - tRVB) < 1e-14
assert lg.norm(tB_SU2.toarray() - tRVB) < 1e-14
assert (tA_SU2.toU1() - tA_U1).norm() < 1e-14
assert (tB_SU2.toU1() - tB_U1).norm() < 1e-14

# check bilayer contraction
a0 = np.tensordot(tRVB, tRVB, ((0, 1), (0, 1)))
a_U1 = tA_U1.H @ tA_U1
a_SU2 = tA_SU2.H @ tA_SU2
assert lg.norm(a_U1.toarray() - a0) < 1e-14
assert lg.norm(a_SU2.toarray() - a0) < 1e-14
assert (a_SU2.toU1() - a_U1).norm() < 1e-14

a1 = a0.transpose(0, 4, 1, 5, 2, 6, 3, 7)
a1_U1 = a_U1.permutate((0, 4, 1, 5), (2, 6, 3, 7))
a1_SU2 = a_SU2.permutate((0, 4, 1, 5), (2, 6, 3, 7))
assert lg.norm(a1_U1.toarray() - a1) < 1e-14
assert lg.norm(a1_SU2.toarray() - a1) < 1e-14
assert (a1_SU2.toU1() - a1_U1).norm() < 1e-14
del a0, a_U1, a_SU2, a1, a1_U1, a1_SU2

# check ctm with AB//BA pattern
ctmU1_AB = CTMRG.from_elementary_tensors("AB\nBA", (tA_U1, tB_U1), 20)
ctmSU2_AB = CTMRG.from_elementary_tensors("AB\nBA", (tA_SU2, tB_SU2), 20)
check_ctm(ctmU1_AB, ctmSU2_AB)


# Now try 1-site unit cell by changing signature on 2 legs.
# Define Sz-reversal operation on virtual leg as
zD = np.eye(D)
zD[1:, 1:] = np.diag(1 - np.arange(d) % 2 * 2)[::-1]


# 1st: UR
print("\n" + "#" * 88)
s = np.array([0, 0, 0, 0, 1, 1])
tRVBur = np.einsum("paurdl,Uu,Rr->paURdl", tRVB, zD, zD)
tRVB_check = tA_SU2.copy()
tRVB_check.update_signature([0, 0, 1, 1, 0, 0])
assert lg.norm(tRVB_check.toarray() - tRVBur) < 1e-14

tur_U1 = U1_SymmetricTensor.from_array(tRVBur, rr_U1, rc_U1, signature=s)
tur_SU2 = SU2_SymmetricTensor.from_array(tRVBur, rr_SU2, rc_SU2, signature=s)
assert lg.norm(tur_U1.toarray() - tRVBur) < 1e-14
assert lg.norm(tur_SU2.toarray() - tRVBur) < 1e-14
assert (tur_SU2 - tRVB_check).norm() < 1e-14
assert (tur_SU2.toU1() - tur_U1).norm() < 1e-14

ctmU1_ur = CTMRG.from_elementary_tensors("A", (tur_U1,), 20)
ctmSU2_ur = CTMRG.from_elementary_tensors("A", (tur_SU2,), 20)
check_ctm(ctmU1_ur, ctmSU2_ur)


# 2nd: RD, with a loop inside signature change
print("\n" + "#" * 88)
s = np.array([0, 0, 1, 0, 0, 1])
tRVBrd = np.einsum("paurdl,rR,Dd->pauRDl", tRVB, zD, zD)
tRVB_check = tA_SU2.copy()
tRVB_check.update_signature([0, 0, 0, -1, 1, 0])
assert lg.norm(tRVB_check.toarray() - tRVBrd) < 1e-14
trd_U1 = U1_SymmetricTensor.from_array(tRVBrd, rr_U1, rc_U1, signature=s)
trd_SU2 = SU2_SymmetricTensor.from_array(tRVBrd, rr_SU2, rc_SU2, signature=s)
assert (trd_SU2.toU1() - trd_U1).norm() < 1e-14
ctmU1_rd = CTMRG.from_elementary_tensors("A", (trd_U1,), 20)
ctmSU2_rd = CTMRG.from_elementary_tensors("A", (trd_SU2,), 20)
check_ctm(ctmU1_rd, ctmSU2_rd)


# 3rd: DL
print("\n" + "#" * 88)
s = np.array([0, 0, 1, 1, 0, 0])
tRVBdl = np.einsum("paurdl,Dd,Ll->paurDL", tRVB, zD, zD)
tRVB_check = tA_SU2.copy()
tRVB_check.update_signature([0, 0, 0, 0, 1, 1])
assert lg.norm(tRVB_check.toarray() - tRVBdl) < 1e-14
tdl_U1 = U1_SymmetricTensor.from_array(tRVBdl, rr_U1, rc_U1, signature=s)
tdl_SU2 = SU2_SymmetricTensor.from_array(tRVBdl, rr_SU2, rc_SU2, signature=s)
assert (tdl_SU2.toU1() - tdl_U1).norm() < 1e-14
ctmU1_dl = CTMRG.from_elementary_tensors("A", (tdl_U1,), 20)
ctmSU2_dl = CTMRG.from_elementary_tensors("A", (tdl_SU2,), 20)
check_ctm(ctmU1_dl, ctmSU2_dl)
