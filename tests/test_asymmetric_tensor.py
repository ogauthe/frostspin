#!/usr/bin/env python3

import numpy as np

from frostspin import AsymmetricTensor

rng = np.random.default_rng(42)
sht = (10, 11, 12)
nrr = 2
t = rng.normal(size=sht)
row_reps = np.array(sht[:nrr])
col_reps = np.array(sht[nrr:])

st = AsymmetricTensor.from_array(t, row_reps, col_reps)
tm = t.reshape(110, 12)

assert st.shape == sht
assert st.n_row_reps == nrr
assert st.nblocks == 1
assert len(st.blocks) == 1
assert np.allclose(st.blocks[0], tm)

b = rng.normal(size=(10, 11, 2, 2))
stb = AsymmetricTensor.from_array(b, row_reps, (np.array([2]), np.array([2])))
bm = b.reshape(110, 4)

# xm = lg.solve(tm, bm)
# x = xm.reshape(12, 2, 2)
