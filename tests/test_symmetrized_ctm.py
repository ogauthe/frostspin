#!/usr/bin/env python3

import os

import numpy as np

from frostspin import SU2SymmetricTensor
from frostspin.ctmrg import SequentialCTMRG, SimultaneousCTMRG, SymmetrizedCTMRG
from frostspin.simple_update import SimpleUpdate, SymmetrizedSimpleUpdate

ASSERT_TOL = 1e-14
rng = np.random.default_rng(42)


def svdvals(st):
    _, s, _ = st.svd()
    return s


r2 = np.array([[1], [2]])
rd = r2
ra = r2
rDu = np.array([[2, 3], [1, 3]])
rDr = np.array([[3, 3], [1, 3]])
rDd = np.array([[1, 3], [1, 3]])
rDl = np.array([[4, 3], [1, 3]])
chi_target = 50

A = SU2SymmetricTensor.random((rd, ra), (rDu, rDr, rDd, rDl), rng=rng)
A /= A.norm()
B = A.dagger().permute((4, 5), (2, 3, 0, 1))

ctm = SimultaneousCTMRG.from_elementary_tensors("AB\nBA", [A, B], chi_target)
ctm2 = SymmetrizedCTMRG.from_elementary_tensors("AB\nBA", [A, B], chi_target)


for coords in [(0, 0), (0, 1)]:
    assert (ctm.get_A(*coords) - ctm2.get_A(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_T1(*coords) - ctm2.get_T1(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_T2(*coords) - ctm2.get_T2(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_T3(*coords) - ctm2.get_T3(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_T4(*coords) - ctm2.get_T4(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_C1(*coords) - ctm2.get_C1(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_C2(*coords) - ctm2.get_C2(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_C3(*coords) - ctm2.get_C3(*coords)).norm() < ASSERT_TOL
    assert (ctm.get_C4(*coords) - ctm2.get_C4(*coords)).norm() < ASSERT_TOL

    assert (
        ctm.construct_enlarged_dl(*coords) - ctm2.construct_enlarged_dl(*coords)
    ).norm() < ASSERT_TOL
    assert (
        ctm.construct_enlarged_ul(*coords) - ctm2.construct_enlarged_ul(*coords)
    ).norm() < ASSERT_TOL
    assert (
        ctm.construct_enlarged_ur(*coords) - ctm2.construct_enlarged_ur(*coords)
    ).norm() < ASSERT_TOL
    assert (
        ctm.construct_enlarged_dr(*coords) - ctm2.construct_enlarged_dr(*coords)
    ).norm() < ASSERT_TOL

assert np.isclose(ctm.compute_PEPS_norm_log(), ctm2.compute_PEPS_norm_log())
ctm.iterate()
ctm2.iterate()

# due to gauge freedom in SVD, tensors start to differ after any iteration
# gauge independent values like corners singular values should stay the same
# but precision is lower
for coords in [(0, 0), (0, 1)]:
    sdl = svdvals(ctm.construct_enlarged_dl(*coords))
    assert (
        sdl - svdvals(ctm2.construct_enlarged_dl(*coords))
    ).norm() < 100 * ASSERT_TOL * sdl.norm()
    sul = svdvals(ctm.construct_enlarged_ul(*coords))
    assert (
        sul - svdvals(ctm2.construct_enlarged_ul(*coords))
    ).norm() < 100 * ASSERT_TOL * sul.norm()
    sdr = svdvals(ctm.construct_enlarged_dr(*coords))
    assert (
        sdr - svdvals(ctm2.construct_enlarged_dr(*coords))
    ).norm() < 100 * ASSERT_TOL * sdr.norm()
    sur = svdvals(ctm.construct_enlarged_ur(*coords))
    assert (
        sur - svdvals(ctm2.construct_enlarged_ur(*coords))
    ).norm() < 100 * ASSERT_TOL * sur.norm()

    rdm1 = ctm.compute_rdm1x2(*coords)
    rdm2 = ctm2.compute_rdm1x2(*coords)
    assert (rdm1 - rdm2).norm() < ASSERT_TOL

    rdm1 = ctm.compute_rdm2x1(*coords)
    rdm2 = ctm2.compute_rdm2x1(*coords)
    assert (rdm1 - rdm2).norm() < ASSERT_TOL

assert np.isclose(ctm.compute_PEPS_norm_log(), ctm2.compute_PEPS_norm_log())

# reduced density matrices are more well-behaved
for _ in range(4):
    ctm.iterate()
    ctm2.iterate()
    for coords in [(0, 0), (0, 1)]:
        rdm1 = ctm.compute_rdm1x2(*coords)
        rdm2 = ctm2.compute_rdm1x2(*coords)
        assert (rdm1 - rdm2).norm() < ASSERT_TOL

        rdm1 = ctm.compute_rdm2x1(*coords)
        rdm2 = ctm2.compute_rdm2x1(*coords)
        assert (rdm1 - rdm2).norm() < ASSERT_TOL
    assert np.isclose(ctm.compute_PEPS_norm_log(), ctm2.compute_PEPS_norm_log())

# ======================================================================================
sds22 = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)
sds22b = np.array(
    [
        [-0.25, 0.0, 0.0, -0.5],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0],
        [-0.5, 0.0, 0.0, -0.25],
    ]
)

r1 = np.array([[1], [1]])
r3 = np.array([[1], [3]])
r4 = np.array([[1], [4]])
r5 = np.array([[1], [5]])
r6 = np.array([[1], [6]])

r2 = np.array([[1], [2]])
h22 = SU2SymmetricTensor.from_array(sds22.reshape(2, 2, 2, 2), (r2, r2), (r2, r2))
h22b = SU2SymmetricTensor.from_array(
    sds22b.reshape(2, 2, 2, 2), (r2, r2), (r2, r2), [0, 1, 1, 0]
)

tau = 1e-3
D = 7
degen_ratio = 0.9999
rcutoff = 1e-12
chi_target = 49

su0 = SimpleUpdate.square_lattice_first_neighbor(
    h22, D, tau, degen_ratio=degen_ratio, rcutoff=rcutoff
)
su = SymmetrizedSimpleUpdate.from_infinite_temperature(D, tau, [h22b])
assert su.n_bonds == 4
assert su.beta == 0.0
assert su.tau == tau

su.evolve(0.5)
su0.evolve(0.5)
assert su.n_bonds == 4
assert su.beta == 0.5
assert su.tau == tau

reps0 = su0.get_bond_representations()
reps = su.get_bond_representations()
weights0 = su0.get_weights()
weights = su.get_weights()
assert len(reps) == 4
assert len(weights) == 4
for i in range(4):
    assert reps0[i].shape == reps[i].shape
    assert (reps0 == reps[i]).all()
    assert np.allclose(weights0[i], weights[i])

ctm0 = SequentialCTMRG.from_elementary_tensors("AB\nBA", su0.get_tensors(), chi_target)
ctm1 = SimultaneousCTMRG.from_elementary_tensors("AB\nBA", su.get_tensors(), chi_target)
ctm2 = SymmetrizedCTMRG.from_elementary_tensors("AB\nBA", su.get_tensors(), chi_target)

for _ in range(10):
    ctm0.iterate()
    ctm1.iterate()
    ctm2.iterate()

for coords in [(0, 0), (0, 1)]:
    sp0 = ctm0.compute_rdm1x2(*coords).eigh(compute_vectors=False)
    sp1 = ctm1.compute_rdm1x2(*coords).eigh(compute_vectors=False)
    sp2 = ctm2.compute_rdm1x2(*coords).eigh(compute_vectors=False)
    assert (sp0 - sp1).norm() < 1e-7  # different SU: low precision
    assert (sp1 - sp2).norm() < 10 * ASSERT_TOL

    sp0 = ctm0.compute_rdm2x1(*coords).eigh(compute_vectors=False)
    sp1 = ctm1.compute_rdm2x1(*coords).eigh(compute_vectors=False)
    sp2 = ctm2.compute_rdm2x1(*coords).eigh(compute_vectors=False)
    assert (sp0 - sp1).norm() < 1e-7  # different SU: low precision
    assert (sp1 - sp2).norm() < 10 * ASSERT_TOL

rdm0_dr, rdm0_ur = ctm0.compute_rdm_2nd_neighbor_cell()
rdm1_dr, rdm1_ur = ctm1.compute_rdm_2nd_neighbor_cell()
rdm2_dr, rdm2_ur = ctm2.compute_rdm_2nd_neighbor_cell()
assert (rdm0_dr[0] - rdm1_dr[0]).norm() < 1e-7
assert (rdm0_dr[1] - rdm1_dr[1].transpose()).norm() < 1e-7
assert (rdm1_dr[0] - rdm2_dr[0]).norm() < ASSERT_TOL
assert (rdm1_dr[1] - rdm2_dr[1]).norm() < ASSERT_TOL

assert (rdm0_ur[0] - rdm1_ur[0].transpose()).norm() < 1e-7
assert (rdm0_ur[1] - rdm1_ur[1]).norm() < 1e-7
assert (rdm1_ur[0] - rdm2_ur[0]).norm() < ASSERT_TOL
assert (rdm1_ur[1] - rdm2_ur[1]).norm() < ASSERT_TOL


savefile = "save_su_test.npz"
su.save_to_file(savefile)
su2 = SymmetrizedSimpleUpdate.load_from_file(savefile)
assert abs(su2.tau - tau) < 1e-14
assert abs(su2.beta - su.beta) < 1e-14
assert su2.D == su.D
assert su2.Dmax == su.Dmax
assert abs(su2.degen_ratio - su.degen_ratio) < 1e-14
assert abs(su2.rcutoff - su.rcutoff) < 1e-14
assert abs(su2.logZ - su.logZ) < 1e-14

tensors = su.get_tensors()
tensors2 = su2.get_tensors()
assert all((t - t2).norm() < 1e-14 for (t, t2) in zip(tensors, tensors2, strict=True))
assert all((t - t2).norm() < 1e-14 for (t, t2) in zip(tensors, tensors2, strict=True))

su.evolve(1.0)
su0.evolve(1.0)

reps0 = su0.get_bond_representations()
reps = su.get_bond_representations()
weights0 = su0.get_weights()
weights = su.get_weights()
assert len(reps) == 4
assert len(weights) == 4
for i in range(4):
    assert reps0[i].shape == reps[i].shape
    assert (reps0 == reps[i]).all()
    assert np.allclose(weights0[i], weights[i])

ctm0 = SequentialCTMRG.from_elementary_tensors("AB\nBA", su0.get_tensors(), chi_target)
ctm1 = SimultaneousCTMRG.from_elementary_tensors("AB\nBA", su.get_tensors(), chi_target)
ctm2 = SymmetrizedCTMRG.from_elementary_tensors("AB\nBA", su.get_tensors(), chi_target)

for _ in range(10):
    ctm0.iterate()
    ctm1.iterate()
    ctm2.iterate()

assert np.isclose(ctm0.compute_PEPS_norm_log(), ctm1.compute_PEPS_norm_log())
assert np.isclose(ctm1.compute_PEPS_norm_log(), ctm2.compute_PEPS_norm_log())

for coords in [(0, 0), (0, 1)]:
    sp0 = ctm0.compute_rdm1x2(*coords).eigh(compute_vectors=False)
    sp1 = ctm1.compute_rdm1x2(*coords).eigh(compute_vectors=False)
    sp2 = ctm2.compute_rdm1x2(*coords).eigh(compute_vectors=False)
    assert (sp0 - sp1).norm() < 1e-7  # different SU: low precision
    assert (sp1 - sp2).norm() < 10 * ASSERT_TOL

    sp0 = ctm0.compute_rdm2x1(*coords).eigh(compute_vectors=False)
    sp1 = ctm1.compute_rdm2x1(*coords).eigh(compute_vectors=False)
    sp2 = ctm2.compute_rdm2x1(*coords).eigh(compute_vectors=False)
    assert (sp0 - sp1).norm() < 1e-7  # different SU: low precision
    assert (sp1 - sp2).norm() < 10 * ASSERT_TOL

rdm0_dr, rdm0_ur = ctm0.compute_rdm_2nd_neighbor_cell()
rdm1_dr, rdm1_ur = ctm1.compute_rdm_2nd_neighbor_cell()
rdm2_dr, rdm2_ur = ctm2.compute_rdm_2nd_neighbor_cell()
assert (rdm0_dr[0] - rdm1_dr[0]).norm() < 1e-7
assert (rdm0_dr[1] - rdm1_dr[1].transpose()).norm() < 1e-7
assert (rdm1_dr[0] - rdm2_dr[0]).norm() < ASSERT_TOL
assert (rdm1_dr[1] - rdm2_dr[1]).norm() < ASSERT_TOL

assert (rdm0_ur[0] - rdm1_ur[0].transpose()).norm() < 1e-7
assert (rdm0_ur[1] - rdm1_ur[1]).norm() < 1e-7
assert (rdm1_ur[0] - rdm2_ur[0]).norm() < ASSERT_TOL
assert (rdm1_ur[1] - rdm2_ur[1]).norm() < ASSERT_TOL


# ------------------------------------  clean-up  --------------------------------------
os.remove(savefile)
