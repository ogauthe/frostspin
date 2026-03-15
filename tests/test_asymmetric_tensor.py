#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from frostspin import AsymmetricTensor


def test_group_rules():
    r2 = np.array([2])
    r3 = np.array([3])
    r2x3 = AsymmetricTensor.combine_representations([r2, r3], [False, False])
    assert r2x3.shape == (1,)
    assert r2x3.dtype == np.int64
    assert r2x3[0] == 6

    irr = np.array([1])
    irrb = AsymmetricTensor.conjugate_irrep(irr)
    assert irrb.shape == (1,)
    assert irrb.dtype == np.int64
    assert irrb[0] == 1

    rb = AsymmetricTensor.conjugate_irrep(irr)
    assert rb.shape == (1,)
    assert rb.dtype == np.int64
    assert rb[0] == 1


def test_asymmetric_tensor():
    rng = np.random.default_rng(42)
    sht = (10, 11, 12)
    nrr = 1
    t = rng.normal(size=sht)
    row_reps = tuple(np.array([sht[i]]) for i in range(nrr))
    col_reps = tuple(np.array([sht[i]]) for i in range(nrr, len(sht)))

    st = AsymmetricTensor.from_array(t, row_reps, col_reps)
    tm = t.reshape(st.matrix_shape)

    assert st.shape == sht
    assert st.n_row_reps == nrr
    assert st.nblocks == 1
    assert len(st.blocks) == 1
    assert np.allclose(st.blocks[0], tm)
    assert np.allclose(st.toarray(), t)
    assert np.allclose(st.toarray(as_matrix=True), tm)
    assert np.isclose(tm.trace(), st.trace())
    assert st.totrivial() is st


def test_svd():
    rng = np.random.default_rng(42)
    sht = (10, 11, 12)
    nrr = 1
    t = rng.normal(size=sht)
    row_reps = tuple(np.array([sht[i]]) for i in range(nrr))
    col_reps = tuple(np.array([sht[i]]) for i in range(nrr, len(sht)))

    st = AsymmetricTensor.from_array(t, row_reps, col_reps)
    tm = t.reshape(st.matrix_shape)
    _u0, s0, _v0 = lg.svd(tm)
    u, s, v = st.svd()
    assert np.allclose(s.toarray(), s0)
    assert np.allclose((s * s).toarray(), s0 * s0)
    assert np.allclose((2 * s).toarray(), 2 * s0)
    assert np.allclose((1 / s).toarray(), 1 / s0)
    assert np.allclose((s / 2).toarray(), s0 / 2)
    assert (+s) is s
    assert (u * s @ v - st).norm() < 1e-14 * min(st.matrix_shape)
    assert s.ndim == 1
    assert s.signature.shape == (2,)
    assert s.signature.dtype == np.bool
    assert (s.signature == np.array([False, True])).all()
    assert repr(s) == "DiagonalTensor with 1 blocks and Trivial symmetry"

    b = rng.normal(size=(10, 2, 2))
    bst = AsymmetricTensor.from_array(b, row_reps, (np.array([2]), np.array([2])))

    stinv = st.pinv()
    assert np.allclose((st @ stinv).toarray(), np.eye(10))

    xst = stinv @ bst
    assert (st @ xst - bst).norm() < 1e-14


if __name__ == "__main__":
    test_group_rules()
    test_asymmetric_tensor()
    test_svd()
