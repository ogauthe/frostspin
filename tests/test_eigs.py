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


dtype = np.float64
nvals = 20
dmax_full = 10


def get_vals(mat, nvals, dmax_full):
    sig0 = mat.signature[: mat.n_row_reps]
    vals = mat.eigs(
        lambda st: mat @ st,
        mat.row_reps,
        sig0,
        nvals,
        dtype=dtype,
        rng=rng,
        dmax_full=dmax_full,
    )
    return vals


vsu2 = get_vals(mat_su2, nvals, dmax_full)
vo2 = get_vals(mat_o2, nvals, dmax_full)
vu1 = get_vals(mat_u1, nvals, dmax_full)
vas = get_vals(mat_as, nvals, dmax_full)


# due to degen, actual sizes may be larger than nvals
d1 = default[:nvals]
d2 = d1.conj()

assert all((np.abs(d1 - vsu2) < 1e-12) | (np.abs(d2 - vsu2) < 1e-12))
assert all((np.abs(d1 - vo2) < 1e-12) | (np.abs(d2 - vo2) < 1e-12))
assert all((np.abs(d1 - vu1) < 1e-12) | (np.abs(d2 - vu1) < 1e-12))
assert all((np.abs(d1 - vas) < 1e-12) | (np.abs(d2 - vas) < 1e-12))

vals, irreps = mat_su2.eigs(
    lambda st: mat_su2 @ st,
    mat_su2.row_reps,
    mat_su2.signature[:2],
    nvals,
    dtype=dtype,
    rng=rng,
    dmax_full=dmax_full,
    return_dense=False,
)


# pathological case: missing blocks
inds = np.array([1, 2, 4])
blocks = [mat_su2.blocks[i] for i in inds]
block_irreps = mat_su2.block_irreps[inds]
mat = ST_SU2(reps, reps, blocks, block_irreps, mat_su2.signature)
v1 = get_vals(mat, 40, 200)
v2 = get_vals(mat, 4, 2)
