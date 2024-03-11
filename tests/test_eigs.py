#!/usr/bin/env python

import numpy as np
import scipy.linalg as lg

import symmetric_tensor.tools


ST_SU2 = symmetric_tensor.tools.get_symmetric_tensor_type("SU2")
rng = np.random.default_rng(42)
reps = (np.array([[2, 3, 1], [1, 3, 5]]), np.array([[2, 2, 1], [1, 3, 5]]))
mat_su2 = ST_SU2.random(reps, reps, rng=rng)
mat_su2 /= mat_su2.norm()
mat_o2 = mat_su2.toO2()
mat_u1 = mat_su2.toU1()
mat_as = mat_su2.totrivial()
dense = mat_su2.toarray(as_matrix=True)

default = lg.eigvals(dense)
default = default[np.abs(default).argsort()[::-1]]


nvals = 20
dmax_full = 10


vsu2 = mat_su2.eigs(mat_su2, nvals, dmax_full=dmax_full, rng=rng)
vo2 = mat_o2.eigs(mat_o2, nvals, dmax_full=dmax_full, rng=rng)
vu1 = mat_u1.eigs(mat_u1, nvals, dmax_full=dmax_full, rng=rng)
vas = mat_as.eigs(mat_as, nvals, dmax_full=dmax_full, rng=rng)


# due to degen, actual sizes may be larger than nvals
d1 = default[:nvals]
d2 = d1.conj()

assert all((np.abs(d1 - vsu2) < 1e-12) | (np.abs(d2 - vsu2) < 1e-12))
assert all((np.abs(d1 - vo2) < 1e-12) | (np.abs(d2 - vo2) < 1e-12))
assert all((np.abs(d1 - vu1) < 1e-12) | (np.abs(d2 - vu1) < 1e-12))
assert all((np.abs(d1 - vas) < 1e-12) | (np.abs(d2 - vas) < 1e-12))

# test return_dense
vals, irreps = mat_su2.eigs(
    mat_su2,
    nvals,
    rng=rng,
    dmax_full=dmax_full,
    return_dense=False,
)


# test implicit matrix
def matmat(x):
    return mat_su2 @ (mat_su2 @ x)


vals = mat_su2.eigs(
    matmat,
    nvals,
    dmax_full=dmax_full,
    rng=rng,
    reps=mat_su2.row_reps,
    signature=mat_su2.signature[: mat_su2.n_row_reps],
)
assert all((np.abs(d1**2 - vals) < 1e-12) | (np.abs(d2**2 - vals) < 1e-12))


# pathological case: missing blocks
inds = np.array([1, 2, 4])
blocks = [mat_su2.blocks[i] for i in inds]
block_irreps = mat_su2.block_irreps[inds]
mat = ST_SU2(reps, reps, blocks, block_irreps, mat_su2.signature)
v1 = mat.eigs(mat, 40, dmax_full=200, rng=rng)
v2 = mat.eigs(mat, 4, dmax_full=2, rng=rng)
