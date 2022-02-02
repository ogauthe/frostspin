#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from ctmrg.ctmrg import CTMRG

"""
Exhaustive test for CTMRG unit cell. Construct a 4x4 unit cell with all 32 inequivalent
bond differ from each other. Use U(1): each bond has dimension 4, but has different
quantum numbers and contraction with any other bond will fail. Also consider charge
conjugation: conjugate representation does not appear in any other bond.

    0     4     7    10
    |     |     |     |
 3--A--1--B--5--C--8--D--3
    |     |     |     |
    2     6     9    11
    |     |     |     |
14--E-12--F-15--G-17--H-14
    |     |     |     |
   13    16    18    19
    |     |     |     |
22--I-20--J-23--K-25--L-22
    |     |     |     |
   21    24    26    27
    |     |     |     |
29--M-28--N-30--O-31--P-29
    |     |     |     |
    0     4     7    10

"""


def eq_st(st1, st2):
    return (
        st1.match_representations(st2)
        and (st1.block_irreps == st2.block_irreps).all()
        and all((b1 == b2).all() for b1, b2 in zip(st1.blocks, st2.blocks))
    )


# use same physical and ancila for all sites
ap = np.array([-1, 1], dtype=np.int8)
aa = np.array([-1, 1], dtype=np.int8)

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
print(((axes1 == axes1[:, None]).all(axis=2) == np.eye(nx, dtype=bool)).all())
# check no doublet up to sign
print((axes1 == -axes1[:, None]).all(axis=2).sum() == 0)

# bilayer with sign swap
axes2 = (axes1[:, :, None] - axes1[:, None]).reshape(nx, d**2)
print(((axes2 == axes2[:, None]).all(axis=2) == np.eye(nx, dtype=bool)).all())
print((axes2 == -axes2[:, None]).all(axis=2).sum() == 0)

reps = (
    (ap, aa, axes1[0], axes1[1], axes1[2], axes1[3]),
    (-ap, -aa, -axes1[4], -axes1[5], -axes1[6], -axes1[1]),
    (ap, aa, axes1[7], axes1[8], axes1[9], axes1[5]),
    (-ap, -aa, -axes1[10], -axes1[3], -axes1[11], -axes1[8]),
    (-ap, -aa, -axes1[2], -axes1[12], -axes1[13], -axes1[14]),
    (ap, aa, axes1[6], axes1[15], axes1[16], axes1[12]),
    (-ap, -aa, -axes1[9], -axes1[17], -axes1[18], -axes1[15]),
    (ap, aa, axes1[11], axes1[14], axes1[19], axes1[17]),
    (ap, aa, axes1[13], axes1[20], axes1[21], axes1[22]),
    (-ap, -aa, -axes1[16], -axes1[23], -axes1[24], -axes1[20]),
    (ap, aa, axes1[18], axes1[25], axes1[26], axes1[23]),
    (-ap, -aa, -axes1[19], -axes1[22], -axes1[27], -axes1[25]),
    (-ap, -aa, -axes1[21], -axes1[28], -axes1[0], -axes1[29]),
    (ap, aa, axes1[24], axes1[30], axes1[4], axes1[28]),
    (-ap, -aa, -axes1[26], -axes1[31], -axes1[7], -axes1[30]),
    (ap, aa, axes1[27], axes1[29], axes1[10], axes1[31]),
)

rng = np.random.default_rng(42)
t00 = U1_SymmetricTensor.random(reps[0][:2], reps[0][2:], rng=rng)
t01 = U1_SymmetricTensor.random(reps[1][:2], reps[1][2:], rng=rng)
t02 = U1_SymmetricTensor.random(reps[2][:2], reps[2][2:], rng=rng)
t03 = U1_SymmetricTensor.random(reps[3][:2], reps[3][2:], rng=rng)

t10 = U1_SymmetricTensor.random(reps[4][:2], reps[4][2:], rng=rng)
t11 = U1_SymmetricTensor.random(reps[5][:2], reps[5][2:], rng=rng)
t12 = U1_SymmetricTensor.random(reps[6][:2], reps[6][2:], rng=rng)
t13 = U1_SymmetricTensor.random(reps[7][:2], reps[7][2:], rng=rng)

t20 = U1_SymmetricTensor.random(reps[8][:2], reps[8][2:], rng=rng)
t21 = U1_SymmetricTensor.random(reps[9][:2], reps[9][2:], rng=rng)
t22 = U1_SymmetricTensor.random(reps[10][:2], reps[10][2:], rng=rng)
t23 = U1_SymmetricTensor.random(reps[11][:2], reps[11][2:], rng=rng)

t30 = U1_SymmetricTensor.random(reps[12][:2], reps[12][2:], rng=rng)
t31 = U1_SymmetricTensor.random(reps[13][:2], reps[13][2:], rng=rng)
t32 = U1_SymmetricTensor.random(reps[14][:2], reps[14][2:], rng=rng)
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
ctm = CTMRG.from_elementary_tensors(
    tiling,
    tensors,
    13,
    block_chi_ratio=1.2,
    ncv_ratio=2.5,
    cutoff=1e-10,
    degen_ratio=1.0,
    verbosity=2,
)

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
