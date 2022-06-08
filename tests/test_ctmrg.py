#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.asymmetric_tensor import AsymmetricTensor
from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor
from ctmrg.ctmrg import CTMRG


def eq_st(st1, st2):
    return (
        st1.match_representations(st2)
        and (st1.block_irreps == st2.block_irreps).all()
        and all((b1 == b2).all() for b1, b2 in zip(st1.blocks, st2.blocks))
    )


# Consider random tensors with each bond having a different representation of different
# size. CTMRG will crash if any mismatch appears in leg contractions.
rng = np.random.default_rng(42)

rp = np.array([1, -1], dtype=np.int8)
ra = np.array([1, -1, 2], dtype=np.int8)
ru = np.array([1, 1, -1, 0, 2, -2], dtype=np.int8)
rr = np.array([2, 0, 1, -1, 0, 1, -1], dtype=np.int8)
rd = np.array([1, 1, -1, -2], dtype=np.int8)
rl = np.array([1, -2, 1, 0, 0], dtype=np.int8)

tiling = "AB\nBA"
sA = np.array([False, False, True, True, True, True])
sB = np.array([False, False, False, False, False, False])
axesA = (rp, ra, ru, rr, rd, rl)
axesB = (rp, ra, rd, rl, ru, rr)
A_U1 = U1_SymmetricTensor.random(axesA[:2], axesA[2:], rng=rng, signature=sA)
B_U1 = U1_SymmetricTensor.random(axesB[:2], axesB[2:], rng=rng, signature=sB)
axesA = tuple(np.array([t.size]) for t in axesA)
axesB = tuple(np.array([t.size]) for t in axesB)
A_as = AsymmetricTensor.from_array(A_U1.toarray(), axesA[:2], axesA[2:], signature=sA)
B_as = AsymmetricTensor.from_array(B_U1.toarray(), axesB[:2], axesB[2:], signature=sB)

tensorsU1 = (A_U1, B_U1)
tensorsAs = (A_as, B_as)
chi = 20
ctmU1 = CTMRG.from_elementary_tensors(tiling, tensorsU1, chi, verbosity=100)
ctmAs = CTMRG.from_elementary_tensors(tiling, tensorsAs, chi, verbosity=100)

# check rdm before iterating: due to random tensors they do not stay hermitian
rdm2x1_cellU1, rdm1x2_cellU1 = ctmU1.compute_rdm_1st_neighbor_cell()
rdm2x1_cellAs, rdm1x2_cellAs = ctmAs.compute_rdm_1st_neighbor_cell()
rdm_dr_cellU1, rdm_ur_cellU1 = ctmU1.compute_rdm_2nd_neighbor_cell()
rdm_dr_cellAs, rdm_ur_cellAs = ctmAs.compute_rdm_2nd_neighbor_cell()
for i in range(2):
    assert lg.norm(rdm2x1_cellU1[i] - rdm2x1_cellAs[i]) < 1e-13
    assert lg.norm(rdm2x1_cellU1[i] - rdm2x1_cellU1[i].T) < 1e-13
    assert lg.norm(rdm1x2_cellU1[i] - rdm1x2_cellAs[i]) < 1e-13
    assert lg.norm(rdm1x2_cellU1[i] - rdm1x2_cellU1[i].T) < 1e-13
    assert lg.norm(rdm_dr_cellU1[i] - rdm_dr_cellAs[i]) < 1e-13
    assert lg.norm(rdm_dr_cellU1[i] - rdm_dr_cellU1[i].T) < 1e-13
    assert lg.norm(rdm_ur_cellU1[i] - rdm_ur_cellAs[i]) < 1e-13
    assert lg.norm(rdm_ur_cellU1[i] - rdm_ur_cellU1[i].T) < 1e-13

ctmU1.iterate()
ctmU1.iterate()
ctmAs.iterate()
ctmAs.iterate()

rdm2x1_cellU1, rdm1x2_cellU1 = ctmU1.compute_rdm_1st_neighbor_cell()
rdm2x1_cellAs, rdm1x2_cellAs = ctmAs.compute_rdm_1st_neighbor_cell()
rdm_dr_cellU1, rdm_ur_cellU1 = ctmU1.compute_rdm_2nd_neighbor_cell()
rdm_dr_cellAs, rdm_ur_cellAs = ctmAs.compute_rdm_2nd_neighbor_cell()
for i in range(2):  # precision is pretty low
    assert lg.norm(rdm2x1_cellU1[i] - rdm2x1_cellAs[i]) < 2e-4
    assert lg.norm(rdm1x2_cellU1[i] - rdm1x2_cellAs[i]) < 2e-4
    assert lg.norm(rdm_dr_cellU1[i] - rdm_dr_cellAs[i]) < 2e-4
    assert lg.norm(rdm_ur_cellU1[i] - rdm_ur_cellAs[i]) < 2e-4

# check save and load once tensors != init
ctmU1.save_to_file("data_test_ctmrg_U1.npz")
ctm2 = CTMRG.from_file("data_test_ctmrg_U1.npz", verbosity=100)
for (x, y) in ctmU1.neq_coords:
    assert eq_st(ctmU1._env.get_A(x, y), ctm2._env.get_A(x, y))
    assert eq_st(ctmU1._env.get_C1(x, y), ctm2._env.get_C1(x, y))
    assert eq_st(ctmU1._env.get_C2(x, y), ctm2._env.get_C2(x, y))
    assert eq_st(ctmU1._env.get_C3(x, y), ctm2._env.get_C3(x, y))
    assert eq_st(ctmU1._env.get_C4(x, y), ctm2._env.get_C4(x, y))
    assert eq_st(ctmU1._env.get_T1(x, y), ctm2._env.get_T1(x, y))
    assert eq_st(ctmU1._env.get_T2(x, y), ctm2._env.get_T2(x, y))
    assert eq_st(ctmU1._env.get_T3(x, y), ctm2._env.get_T3(x, y))
    assert eq_st(ctmU1._env.get_T4(x, y), ctm2._env.get_T4(x, y))

ctmAs.save_to_file("data_test_ctmrg_As.npz")
ctm2 = CTMRG.from_file("data_test_ctmrg_As.npz", verbosity=100)
for (x, y) in ctmAs.neq_coords:
    assert eq_st(ctmAs._env.get_A(x, y), ctm2._env.get_A(x, y))
    assert eq_st(ctmAs._env.get_C1(x, y), ctm2._env.get_C1(x, y))
    assert eq_st(ctmAs._env.get_C2(x, y), ctm2._env.get_C2(x, y))
    assert eq_st(ctmAs._env.get_C3(x, y), ctm2._env.get_C3(x, y))
    assert eq_st(ctmAs._env.get_C4(x, y), ctm2._env.get_C4(x, y))
    assert eq_st(ctmAs._env.get_T1(x, y), ctm2._env.get_T1(x, y))
    assert eq_st(ctmAs._env.get_T2(x, y), ctm2._env.get_T2(x, y))
    assert eq_st(ctmAs._env.get_T3(x, y), ctm2._env.get_T3(x, y))
    assert eq_st(ctmAs._env.get_T4(x, y), ctm2._env.get_T4(x, y))


########################################################################################
# Test CTMRG on RVB SU(2) wavefunction
########################################################################################
# current implementation fails with half-integer spin
d = 2
D = d + 1

tRVB = np.zeros((d, 1, D, D, D, D))
for i in range(d):
    tRVB[i, 0, 0, 0, 0, i + 1] = 1.0
    tRVB[i, 0, 0, 0, i + 1, 0] = 1.0
    tRVB[i, 0, 0, i + 1, 0, 0] = 1.0
    tRVB[i, 0, i + 1, 0, 0, 0] = 1.0

rep_d_asym = np.array([d])
rep_a_asym = np.array([1])
rep_D_asym = np.array([D])

rep_d_U1 = np.arange(d - 1, -d, -2, dtype=np.int8)
rep_a_U1 = np.array([0], dtype=np.int8)
rep_D_U1 = np.array([0, *rep_d_U1], dtype=np.int8)

rep_d_SU2 = np.array([[1], [d]])
rep_a_SU2 = np.array([[1], [1]])
rep_D_SU2 = np.array([[1, 1], [1, d]])

# we need to reverse arrows to use 1-site unit cell
zD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
tRVBzz = np.einsum("paurdl,dD,lL->paurDL", tRVB, zD, zD)

s = np.array([False, False, True, True, False, False])
tRVB_asym = AsymmetricTensor.from_array(
    tRVBzz, (rep_d_asym, rep_a_asym), (rep_D_asym,) * 4, signature=s
)
tRVB_U1 = U1_SymmetricTensor.from_array(
    tRVBzz, (rep_d_U1, rep_a_U1), (rep_D_U1, rep_D_U1, rep_D_U1, rep_D_U1), signature=s
)
tRVB_SU2 = SU2_SymmetricTensor.from_array(
    tRVBzz, (rep_d_SU2, rep_a_SU2), (rep_D_SU2,) * 4, signature=s
)

assert lg.norm(tRVB_asym.toarray() - tRVBzz) < 1e-13
assert lg.norm(tRVB_U1.toarray() - tRVBzz) < 1e-13
assert lg.norm(tRVB_SU2.toarray() - tRVBzz) < 1e-13

a0 = np.tensordot(tRVBzz, tRVBzz, ((0, 1), (0, 1)))
a_asym = tRVB_asym.H @ tRVB_asym
a_U1 = tRVB_U1.H @ tRVB_U1
a_SU2 = tRVB_SU2.H @ tRVB_SU2
assert lg.norm(a_asym.toarray() - a0) < 1e-13
assert lg.norm(a_U1.toarray() - a0) < 1e-13
assert lg.norm(a_SU2.toarray() - a0) < 1e-13

a1 = a0.transpose(0, 4, 1, 5, 2, 6, 3, 7)
a1_asym = a_asym.permutate((0, 4, 1, 5), (2, 6, 3, 7))
a1_U1 = a_U1.permutate((0, 4, 1, 5), (2, 6, 3, 7))
a1_SU2 = a_SU2.permutate((0, 4, 1, 5), (2, 6, 3, 7))
assert lg.norm(a1_asym.toarray() - a1) < 1e-13
assert lg.norm(a1_U1.toarray() - a1) < 1e-13
assert lg.norm(a1_SU2.toarray() - a1) < 1e-13
del a0, a_asym, a_U1, a_SU2, a1, a1_asym, a1_U1, a1_SU2

ctmAs = CTMRG.from_elementary_tensors("A", (tRVB_asym,), 20, verbosity=100)
ctmU1 = CTMRG.from_elementary_tensors("A", (tRVB_U1,), 20, verbosity=100)

# something is wrong when mixing integer and half-integer spins
# ctmSU2 = CTMRG.from_elementary_tensors("A", (tRVB_SU2,), 20, verbosity=100)

rdmAs = ctmAs.compute_rdm2x1(0, 0)
rdmU1 = ctmU1.compute_rdm2x1(0, 0)
# rdmSU2 = ctmSU2.compute_rdm2x1(0, 0)
print(lg.eigvalsh(rdmAs), f" {lg.norm(rdmAs-rdmAs.T.conj()):.0e}")
print(lg.eigvalsh(rdmU1), f" {lg.norm(rdmU1-rdmU1.T.conj()):.0e}")
# print(lg.eigvalsh(rdmSU2), f" {lg.norm(rdmSU2-rdmSU2.T.conj()):.0e}")

print("Completed")
