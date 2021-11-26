#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

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
A0 = U1_SymmetricTensor.random(axesA[:2], axesA[2:], rng=rng).toarray()
axesB = (-rp, -ra, -rd, -rl, -ru, -rr)
B0 = U1_SymmetricTensor.random(axesB[:2], axesB[2:], rng=rng).toarray()

tensors = (A0, B0)
reps = (axesA, axesB)
chi = 20
ctm = CTMRG.from_elementary_tensors(tiling, tensors, reps, chi, verbosity=100)

# check rdm before iterating: due to random tensors they do not stay hermitian
rdm2x1_cell, rdm1x2_cell = ctm.compute_rdm_1st_neighbor_cell()
for m in rdm2x1_cell:
    assert lg.norm(m - m.T) < 1e-13
for m in rdm1x2_cell:
    assert lg.norm(m - m.T) < 1e-13

rdm_dr_cell, rdm_ur_cell = ctm.compute_rdm_2nd_neighbor_cell()
for m in rdm_dr_cell:
    assert lg.norm(m - m.T) < 1e-13
for m in rdm_ur_cell:
    assert lg.norm(m - m.T) < 1e-13

ctm.iterate()
ctm.iterate()

# check save and load once tensors != init
ctm.save_to_file("data_test_ctmrg.npz")
ctm2 = CTMRG.from_file("data_test_ctmrg.npz", verbosity=100)
for (x, y) in ctm.neq_coords:
    assert eq_st(ctm._env.get_A(x, y), ctm2._env.get_A(x, y))
    assert eq_st(ctm._env.get_C1(x, y), ctm2._env.get_C1(x, y))
    assert eq_st(ctm._env.get_C2(x, y), ctm2._env.get_C2(x, y))
    assert eq_st(ctm._env.get_C3(x, y), ctm2._env.get_C3(x, y))
    assert eq_st(ctm._env.get_C4(x, y), ctm2._env.get_C4(x, y))
    assert eq_st(ctm._env.get_T1(x, y), ctm2._env.get_T1(x, y))
    assert eq_st(ctm._env.get_T2(x, y), ctm2._env.get_T2(x, y))
    assert eq_st(ctm._env.get_T3(x, y), ctm2._env.get_T3(x, y))
    assert eq_st(ctm._env.get_T4(x, y), ctm2._env.get_T4(x, y))

print("Completed")
