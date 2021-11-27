#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.asymmetric_tensor import AsymmetricTensor
from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
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
axesA = (rp, ra, ru, rr, rd, rl)
axesB = (-rp, -ra, -rd, -rl, -ru, -rr)
A_U1 = U1_SymmetricTensor.random(axesA[:2], axesA[2:], rng=rng)
B_U1 = U1_SymmetricTensor.random(axesB[:2], axesB[2:], rng=rng)
axesA = tuple(np.array(t.size) for t in axesA)
axesB = tuple(np.array(t.size) for t in axesB)
A_as = AsymmetricTensor.from_array(A_U1.toarray(), axesA[:2], axesA[2:])
B_as = AsymmetricTensor.from_array(B_U1.toarray(), axesB[:2], axesB[2:])

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
    assert lg.norm(rdm2x1_cellU1[i] - rdm2x1_cellAs[i]) < 3e-5
    assert lg.norm(rdm1x2_cellU1[i] - rdm1x2_cellAs[i]) < 3e-5
    assert lg.norm(rdm_dr_cellU1[i] - rdm_dr_cellAs[i]) < 3e-5
    assert lg.norm(rdm_ur_cellU1[i] - rdm_ur_cellAs[i]) < 3e-5

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

print("Completed")
