#!/usr/bin/env python3

import os

import numpy as np
import scipy.linalg as lg

from froSTspin.ctmrg.ctmrg import CTMRG
from froSTspin.symmetric_tensor.asymmetric_tensor import AsymmetricTensor
from froSTspin.symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor

# try simplest CTMRG and parameter parsing
a = AsymmetricTensor.from_array(
    np.ones((2, 2, 3, 3, 3, 3)), np.array([[2, 2]]).T, np.array([[3, 3, 3, 3]]).T
)
b = a.dual()
ctm0 = CTMRG.from_elementary_tensors(
    "AB\nBA",
    [a, b],
    20,
    block_chi_ratio=1.2,
    block_ncv_ratio=2.2,
    cutoff=1e-10,
    degen_ratio=1.0,
)

assert ctm0.Dmax == 3
assert ctm0.Dmin == 3
assert ctm0.Lx == 2
assert ctm0.Ly == 2
assert ctm0.cell.shape == (2, 2)
assert (ctm0.cell == np.array([["A", "B"], ["B", "A"]])).all()
assert ctm0.chi_target == 20
assert ctm0.chi_values == [1]
assert len(ctm0.elementary_tensors) == 2
assert ctm0.elementary_tensors[0] is a
assert ctm0.elementary_tensors[1] is b
assert ctm0.n_sites == 2
assert ctm0.site_coords.shape == (2, 2)
assert (ctm0.site_coords == np.array([[0, 0], [1, 0]])).all()
assert ctm0.tiling == "AB\nBA"
assert ctm0.symmetry() == "trivial"
assert len(ctm0.get_corner_representations()) == 1
assert len(ctm0.get_corner_representations()) == 1
assert (ctm0.get_corner_representations()[0] == a.singlet()).all()


def eq_st(st1, st2):
    return (
        st1.match_representations(st2)
        and (st1.block_irreps == st2.block_irreps).all()
        and all((b1 == b2).all() for b1, b2 in zip(st1.blocks, st2.blocks, strict=True))
    )


# Consider random tensors with each bond having a different representation of different
# size. CTMRG will crash if any mismatch appears in leg contractions.
rng = np.random.default_rng(42)
savefile = "data_test_ctmrg.npz"


rp = np.array([1, -1], dtype=np.int8)
ra = np.array([1, -1, 2], dtype=np.int8)
ru = np.array([1, 1, -1, 0, 2, -2], dtype=np.int8)
rr = np.array([2, 0, 1, -1, 0, 1, -1], dtype=np.int8)
rd = np.array([1, 1, -1, -2], dtype=np.int8)
rl = np.array([1, -2, 1, 0, 0], dtype=np.int8)

tiling = "AB\nBA"
sA = np.array([False, False, True, True, True, True])
sB = np.array([False, False, False, False, False, False])
reps_U1_A = (rp, ra, ru, rr, rd, rl)
reps_U1_B = (rp, ra, rd, rl, ru, rr)
A_U1 = U1_SymmetricTensor.random(reps_U1_A[:2], reps_U1_A[2:], rng=rng, signature=sA)
B_U1 = U1_SymmetricTensor.random(reps_U1_B[:2], reps_U1_B[2:], rng=rng, signature=sB)
reps_as_A = tuple(np.array([t.size]) for t in reps_U1_A)
reps_as_B = tuple(np.array([t.size]) for t in reps_U1_B)
A_as = AsymmetricTensor.from_array(
    A_U1.toarray(), reps_as_A[:2], reps_as_A[2:], signature=sA
)
B_as = AsymmetricTensor.from_array(
    B_U1.toarray(), reps_as_B[:2], reps_as_B[2:], signature=sB
)

tensorsAs = (A_as, B_as)
tensorsU1 = (A_U1, B_U1)
chi = 20

ctmAs = CTMRG.from_elementary_tensors(tiling, tensorsAs, chi)
ctmU1 = CTMRG.from_elementary_tensors(tiling, tensorsU1, chi)

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

# check xi computation succeeds
xih1_U1 = ctmU1.compute_corr_length_h(y=0)
xih2_U1 = ctmU1.compute_corr_length_h(y=1)
xiv1_U1 = ctmU1.compute_corr_length_v(x=0)
xiv2_U1 = ctmU1.compute_corr_length_v(x=1)
xih1_As = ctmAs.compute_corr_length_h(y=0)
xih2_As = ctmAs.compute_corr_length_h(y=1)
xiv1_As = ctmAs.compute_corr_length_v(x=0)
xiv2_As = ctmAs.compute_corr_length_v(x=1)

# with AB//BA pattern, xih1 and xih2 should be nearly the same
# eigs precision is limited, set high tol for comparison U(1) / trivial
assert xih1_U1 > 0
assert abs(xih2_U1 - xih1_U1) < 1e-10 * xih1_U1
assert abs(xih1_As - xih1_U1) < 0.02 * xih1_U1
assert abs(xih2_As - xih1_U1) < 0.02 * xih1_U1
assert xiv1_U1 > 0
assert abs(xiv2_U1 - xiv1_U1) < 1e-10 * xiv1_U1
assert abs(xiv1_As - xiv1_U1) < 0.02 * xih1_U1
assert abs(xiv2_As - xiv1_U1) < 0.02 * xih1_U1

# check save and load once tensors != init
ctmU1.save_to_file(savefile)
ctm2 = CTMRG.load_from_file(savefile)
for x, y in ctmU1.site_coords:
    assert eq_st(ctmU1.get_A(x, y), ctm2.get_A(x, y))
    assert eq_st(ctmU1.get_C1(x, y), ctm2.get_C1(x, y))
    assert eq_st(ctmU1.get_C2(x, y), ctm2.get_C2(x, y))
    assert eq_st(ctmU1.get_C3(x, y), ctm2.get_C3(x, y))
    assert eq_st(ctmU1.get_C4(x, y), ctm2.get_C4(x, y))
    assert eq_st(ctmU1.get_T1(x, y), ctm2.get_T1(x, y))
    assert eq_st(ctmU1.get_T2(x, y), ctm2.get_T2(x, y))
    assert eq_st(ctmU1.get_T3(x, y), ctm2.get_T3(x, y))
    assert eq_st(ctmU1.get_T4(x, y), ctm2.get_T4(x, y))

ctmAs.save_to_file(savefile)
ctm2 = CTMRG.load_from_file(savefile)
for x, y in ctmAs.site_coords:
    assert eq_st(ctmAs.get_A(x, y), ctm2.get_A(x, y))
    assert eq_st(ctmAs.get_C1(x, y), ctm2.get_C1(x, y))
    assert eq_st(ctmAs.get_C2(x, y), ctm2.get_C2(x, y))
    assert eq_st(ctmAs.get_C3(x, y), ctm2.get_C3(x, y))
    assert eq_st(ctmAs.get_C4(x, y), ctm2.get_C4(x, y))
    assert eq_st(ctmAs.get_T1(x, y), ctm2.get_T1(x, y))
    assert eq_st(ctmAs.get_T2(x, y), ctm2.get_T2(x, y))
    assert eq_st(ctmAs.get_T3(x, y), ctm2.get_T3(x, y))
    assert eq_st(ctmAs.get_T4(x, y), ctm2.get_T4(x, y))


########################################################################################
#
# Exhaustive test for CTMRG unit cell. Construct a 4x4 unit cell with all 32
# inequivalent bond differ from each other. Use U(1): each bond has dimension 4, but has
# different quantum numbers and contraction with any other bond will fail. Also consider
# charge conjugation: conjugate representation does not appear in any other bond.
#
#     0     4     7    10
#     |     |     |     |
#  3--A--1--B--5--C--8--D--3
#     |     |     |     |
#     2     6     9    11
#     |     |     |     |
# 14--E-12--F-15--G-17--H-14
#     |     |     |     |
#    13    16    18    19
#     |     |     |     |
# 22--I-20--J-23--K-25--L-22
#     |     |     |     |
#    21    24    26    27
#     |     |     |     |
# 29--M-28--N-30--O-31--P-29
#     |     |     |     |
#     0     4     7    10


# use same physical and ancila for all sites
ap = np.array([1, -1], dtype=np.int8)
aa = np.array([1, -1], dtype=np.int8)

axes1 = np.array(
    [
        [0, 0, -1, 1],
        [0, 0, -2, 2],
        [0, 1, -1, 0],
        [0, 1, 0, -1],
        [0, 2, -2, 0],
        [0, 2, 0, -2],
        [-1, 0, 0, 1],
        [-1, 0, 1, 0],
        [-1, 1, 0, 0],
        [-1, -1, 1, 1],
        [-1, 1, -1, 1],
        [-1, 1, -2, 2],
        [-1, 1, 2, -2],
        [-1, -2, 2, 1],
        [-1, 2, -2, 1],
        [-1, -2, 1, 2],
        [-1, 2, 1, -2],
        [-2, 0, 0, 2],
        [-2, 0, 2, 0],
        [-2, -1, 1, 2],
        [-2, 1, -1, 2],
        [-2, -1, 2, 1],
        [-2, 1, 2, -1],
        [-2, 2, 0, 0],
        [-2, -2, 2, 2],
        [-2, 2, -2, 2],
        [-2, 2, -1, 1],
        [-2, 2, 1, -1],
        [-3, 0, 0, 3],
        [-3, 0, 3, 0],
        [-3, -1, 1, 3],
        [-3, 1, -1, 3],
        [-3, -1, 3, 1],
        [-3, 1, 3, -1],
        [-3, 3, 0, 0],
        [-3, -3, 3, 3],
        [-3, 3, -3, 3],
        [-3, 3, -1, 1],
        [-3, 3, 1, -1],
    ],
    dtype=np.int8,
)

nx = axes1.shape[0]
d = axes1.shape[1]

# check no doublet in axes1
assert ((axes1 == axes1[:, None]).all(axis=2) == np.eye(nx, dtype=bool)).all()
# check no doublet up to sign
assert (axes1 == -axes1[:, None]).all(axis=2).sum() == 0

# bilayer with sign swap
axes2 = (axes1[:, :, None] - axes1[:, None]).reshape(nx, d**2)
assert ((axes2 == axes2[:, None]).all(axis=2) == np.eye(nx, dtype=bool)).all()
assert (axes2 == -axes2[:, None]).all(axis=2).sum() == 0

reps = (
    (ap, aa, axes1[0], axes1[1], axes1[2], axes1[3]),
    (ap, aa, axes1[4], axes1[5], axes1[6], axes1[1]),
    (ap, aa, axes1[7], axes1[8], axes1[9], axes1[5]),
    (ap, aa, axes1[10], axes1[3], axes1[11], axes1[8]),
    (ap, aa, axes1[2], axes1[12], axes1[13], axes1[14]),
    (ap, aa, axes1[6], axes1[15], axes1[16], axes1[12]),
    (ap, aa, axes1[9], axes1[17], axes1[18], axes1[15]),
    (ap, aa, axes1[11], axes1[14], axes1[19], axes1[17]),
    (ap, aa, axes1[13], axes1[20], axes1[21], axes1[22]),
    (ap, aa, axes1[16], axes1[23], axes1[24], axes1[20]),
    (ap, aa, axes1[18], axes1[25], axes1[26], axes1[23]),
    (ap, aa, axes1[19], axes1[22], axes1[27], axes1[25]),
    (ap, aa, axes1[21], axes1[28], axes1[0], axes1[29]),
    (ap, aa, axes1[24], axes1[30], axes1[4], axes1[28]),
    (ap, aa, axes1[26], axes1[31], axes1[7], axes1[30]),
    (ap, aa, axes1[27], axes1[29], axes1[10], axes1[31]),
)

rng = np.random.default_rng(42)

# signature on sublattice B
sB = np.array([True, True, False, False, False, False])
t00 = U1_SymmetricTensor.random(reps[0][:2], reps[0][2:], rng=rng)
t01 = U1_SymmetricTensor.random(reps[1][:2], reps[1][2:], rng=rng, signature=sB)
t02 = U1_SymmetricTensor.random(reps[2][:2], reps[2][2:], rng=rng)
t03 = U1_SymmetricTensor.random(reps[3][:2], reps[3][2:], rng=rng, signature=sB)

t10 = U1_SymmetricTensor.random(reps[4][:2], reps[4][2:], rng=rng, signature=sB)
t11 = U1_SymmetricTensor.random(reps[5][:2], reps[5][2:], rng=rng)
t12 = U1_SymmetricTensor.random(reps[6][:2], reps[6][2:], rng=rng, signature=sB)
t13 = U1_SymmetricTensor.random(reps[7][:2], reps[7][2:], rng=rng)

t20 = U1_SymmetricTensor.random(reps[8][:2], reps[8][2:], rng=rng)
t21 = U1_SymmetricTensor.random(reps[9][:2], reps[9][2:], rng=rng, signature=sB)
t22 = U1_SymmetricTensor.random(reps[10][:2], reps[10][2:], rng=rng)
t23 = U1_SymmetricTensor.random(reps[11][:2], reps[11][2:], rng=rng, signature=sB)

t30 = U1_SymmetricTensor.random(reps[12][:2], reps[12][2:], rng=rng, signature=sB)
t31 = U1_SymmetricTensor.random(reps[13][:2], reps[13][2:], rng=rng)
t32 = U1_SymmetricTensor.random(reps[14][:2], reps[14][2:], rng=rng, signature=sB)
t33 = U1_SymmetricTensor.random(reps[15][:2], reps[15][2:], rng=rng)

tensors = (
    t00,
    t01,
    t02,
    t03,
    t10,
    t11,
    t12,
    t13,
    t20,
    t21,
    t22,
    t23,
    t30,
    t31,
    t32,
    t33,
)

tiling = "ABCD\nEFGH\nIJKL\nMNOP"
ctm = CTMRG.from_elementary_tensors(tiling, tensors, 13)

rdm2x1_cell, rdm1x2_cell = ctm.compute_rdm_1st_neighbor_cell()
for m in rdm2x1_cell:
    assert lg.norm(m - m.T) < 1e-8
for m in rdm1x2_cell:
    assert lg.norm(m - m.T) < 1e-8

rdm_dr_cell, rdm_ur_cell = ctm.compute_rdm_2nd_neighbor_cell()
for m in rdm_dr_cell:
    assert lg.norm(m - m.T) < 1e-8
for m in rdm_ur_cell:
    assert lg.norm(m - m.T) < 1e-8

ctm.iterate()
ctm.iterate()

# check truncate_corners succeeds
ctm.restart_environment()
ctm.truncate_corners()
ctm.iterate()
ctm.iterate()

# after iterate, rdm are not really hermitian, precision is around 1e-3
# due to U(1), measure is made on 1 coeff only
# do not bother check for it, just check computations succeeds
rdm2x1_cell, rdm1x2_cell = ctm.compute_rdm_1st_neighbor_cell()
rdm_dr_cell, rdm_ur_cell = ctm.compute_rdm_2nd_neighbor_cell()

# check save and load once tensors != init
ctm.save_to_file(savefile)
ctm2 = CTMRG.load_from_file(savefile)
for x, y in ctm.site_coords:
    assert eq_st(ctm.get_A(x, y), ctm2.get_A(x, y))
    assert eq_st(ctm.get_C1(x, y), ctm2.get_C1(x, y))
    assert eq_st(ctm.get_C2(x, y), ctm2.get_C2(x, y))
    assert eq_st(ctm.get_C3(x, y), ctm2.get_C3(x, y))
    assert eq_st(ctm.get_C4(x, y), ctm2.get_C4(x, y))
    assert eq_st(ctm.get_T1(x, y), ctm2.get_T1(x, y))
    assert eq_st(ctm.get_T2(x, y), ctm2.get_T2(x, y))
    assert eq_st(ctm.get_T3(x, y), ctm2.get_T3(x, y))
    assert eq_st(ctm.get_T4(x, y), ctm2.get_T4(x, y))

os.remove(savefile)


# check dummy=False environment initialization
# TODO
"""
ctm = CTMRG.from_elementary_tensors(tiling, tensors, 13, dummy=False)

rdm2x1_cell, rdm1x2_cell = ctm.compute_rdm_1st_neighbor_cell()
for m in rdm2x1_cell:
    assert lg.norm(m - m.T) < 1e-8
for m in rdm1x2_cell:
    assert lg.norm(m - m.T) < 1e-8

rdm_dr_cell, rdm_ur_cell = ctm.compute_rdm_2nd_neighbor_cell()
for m in rdm_dr_cell:
    assert lg.norm(m - m.T) < 1e-8
for m in rdm_ur_cell:
    assert lg.norm(m - m.T) < 1e-8

ctm.iterate()
ctm.iterate()

ctm.restart_environment(dummy=True)
ctm.truncate_corners()
ctm.iterate()
ctm.iterate()
rdm2x1_cell, rdm1x2_cell = ctm.compute_rdm_1st_neighbor_cell()
rdm_dr_cell, rdm_ur_cell = ctm.compute_rdm_2nd_neighbor_cell()
"""
