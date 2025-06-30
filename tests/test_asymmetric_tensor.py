#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from frostspin import AsymmetricTensor

rng = np.random.default_rng(42)
sht = (10, 11, 12)
nrr = 1
t = rng.normal(size=sht)
row_reps = np.array(sht[:nrr])
col_reps = np.array(sht[nrr:])

st = AsymmetricTensor.from_array(t, row_reps, col_reps)
tm = t.reshape(st.matrix_shape)

assert st.shape == sht
assert st.n_row_reps == nrr
assert st.nblocks == 1
assert len(st.blocks) == 1
assert np.allclose(st.blocks[0], tm)
assert np.allclose(st.toarray(), t)
assert np.allclose(st.toarray(as_matrix=True), tm)

u0, s0, v0 = lg.svd(tm)
u, s, v = st.svd()
assert np.allclose(s.toarray(), s0)
assert np.allclose((s * s).toarray(), s0 * s0)
assert np.allclose((2 * s).toarray(), 2 * s0)
assert np.allclose((1 / s).toarray(), 1 / s0)
assert (u * s @ v - st).norm() < 1e-14 * min(st.matrix_shape)

b = rng.normal(size=(10, 2, 2))
bst = AsymmetricTensor.from_array(b, row_reps, (np.array([2]), np.array([2])))
bm = b.reshape(bst.matrix_shape)

stinv = st.pinv()
assert np.allclose((st @ stinv).toarray(), np.eye(10))

xst = stinv @ bst
assert (st @ xst - bst).norm() < 1e-14
