#!/usr/bin/env python

import os
import tempfile

import numpy as np

from frostspin import U1SymmetricTensor
from frostspin.simple_update import SimpleUpdate

D = 4
tau = 0.02
degen_ratio = 0.99
rcutoff = 1e-8
sds = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)
sds = sds.reshape(2, 2, 2, 2)

reps = (np.array([1, -1], dtype=np.int8),) * 2
h1 = U1SymmetricTensor.from_array(sds, reps, reps)


def test_simple_update_1stnei(tmp_path):
    su = SimpleUpdate.square_lattice_first_neighbor(
        h1, 7, 0.01, degen_ratio=degen_ratio, rcutoff=rcutoff
    )
    assert su.tau == 0.01
    assert su.D == 7
    assert su.Dmax == 1
    assert su.beta == 0.0
    assert su.degen_ratio == degen_ratio
    assert su.rcutoff == rcutoff
    assert su.logZ == 0.0
    assert su.n_bonds == 4
    assert su.n_tensors == 2
    assert su.symmetry() == "U1"

    su.tau = tau
    assert su.tau == tau
    su.D = D
    assert su.D == D

    beta = 4 * tau
    su.evolve(beta)
    assert su.Dmax == D
    assert abs(su.beta - beta) < 1e-14
    assert su.logZ > 0.0

    bond_reps = su.get_bond_representations()
    assert all((r == np.array([-2, 0, 0, 2])).all() for r in bond_reps)
    weights = su.get_weights()
    assert len(weights) == 4
    tensors = su.get_tensors()
    assert len(tensors) == 2

    savefile = os.path.join(tmp_path, "save.npz")
    su.save_to_file(savefile)
    su2 = SimpleUpdate.load_from_file(savefile)
    assert abs(su2.tau - tau) < 1e-14
    assert abs(su2.beta - su.beta) < 1e-14
    assert su2.D == D
    assert su2.Dmax == su.Dmax
    assert abs(su2.degen_ratio - su.degen_ratio) < 1e-14
    assert abs(su2.rcutoff - su.rcutoff) < 1e-14
    assert abs(su2.logZ - su.logZ) < 1e-14

    tensors2 = su2.get_tensors()
    assert all(
        (t - t2).norm() < 1e-14 for (t, t2) in zip(tensors, tensors2, strict=True)
    )


def test_simple_update_2nd_neighbor():
    su = SimpleUpdate.square_lattice_second_neighbor(
        h1, h1, D, tau, degen_ratio=degen_ratio, rcutoff=rcutoff
    )

    assert su.tau == tau
    assert su.D == D
    assert su.Dmax == 1
    assert su.beta == 0.0
    assert su.degen_ratio == degen_ratio
    assert su.rcutoff == rcutoff
    assert su.logZ == 0.0
    assert su.n_bonds == 8
    assert su.n_tensors == 4
    assert su.symmetry() == "U1"

    beta = 4 * tau
    su.evolve(beta)
    assert su.Dmax == D
    assert abs(su.beta - beta) < 1e-14
    assert su.logZ > 0.0

    bond_reps = su.get_bond_representations()
    assert all((r == np.array([-2, 0, 0, 2])).all() for r in bond_reps)
    weights = su.get_weights()
    assert len(weights) == 8
    tensors = su.get_tensors()
    assert len(tensors) == 4


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_path:
        test_simple_update_1stnei(tmp_path)
    test_simple_update_2nd_neighbor()
