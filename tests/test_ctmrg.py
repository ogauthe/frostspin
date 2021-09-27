#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from ctmrg.ctmrg import CTMRG_U1


def random_U1_tensor(axis_reps, rng=None):
    """
    Construct random U(1) symmetric tensor. Non-zero coefficients are taken from
    continuous uniform distribution in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    axis_reps : enumerable of 1D integer arrays.
        U(1) quantum numbers of each axis.
    rng : optional, random number generator. Can be used to reproduce results.

    Returns
    -------
    output : ndarray
        random U(1) tensor, with shape following axis_reps dimensons.
    """
    if rng is None:
        rng = np.random.default_rng()
    irreps1D = U1_SymmetricTensor.combine_representations(*axis_reps)
    nnz = (irreps1D == 0).nonzero()[0]
    t0 = np.zeros(irreps1D.size)
    t0[nnz] = rng.random(nnz.size)
    t0 = t0.reshape(tuple(r.size for r in axis_reps))
    return t0


def eq_st(st1, st2):
    if type(st1) != type(st2):
        return False
    if st1.shape != st2.shape:
        return False
    if st1.nblocks != st2.nblocks:
        return False
    for ax in range(st1.ndim):
        if not np.asarray(st1.axis_reps[ax] == st2.axis_reps[ax]).all():
            return False
    for bi in range(st1.nblocks):
        if not (st1.blocks[bi] == st2.blocks[bi]).all():
            return False
    return True


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
A0 = random_U1_tensor(axesA, rng)
axesB = (-rp, -ra, -rd, -rl, -ru, -rr)
B0 = random_U1_tensor(axesB, rng)

tensors = (A0, B0)
colors = (axesA, axesB)
chi = 20
ctm = CTMRG_U1.from_elementary_tensors(tensors, colors, tiling, chi, verbosity=100)

ctm.save_to_file("data_test_ctmrg.npz")
ctm2 = CTMRG_U1.from_file("data_test_ctmrg.npz")
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
