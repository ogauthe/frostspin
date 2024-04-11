#!/usr/bin/env python

import numpy as np
import scipy.linalg as lg

import froSTspin.symmetric_tensor.tools

ST_SU2 = froSTspin.symmetric_tensor.tools.get_symmetric_tensor_type("SU2")
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
assert vsu2.dtype == np.complex128

vsu2 = vsu2.toarray(sort=True)[:nvals]
vo2 = mat_o2.eigs(mat_o2, nvals, dmax_full=dmax_full, rng=rng)
vo2 = vo2.toarray(sort=True)[:nvals]
vu1 = mat_u1.eigs(mat_u1, nvals, dmax_full=dmax_full, rng=rng)
vu1 = vu1.toarray(sort=True)[:nvals]
vas = mat_as.eigs(mat_as, nvals, dmax_full=dmax_full, rng=rng)
vas = vas.toarray()[:nvals]


# due to degen, actual sizes may be larger than nvals
d1 = default[:nvals]
d2 = d1.conj()

assert all((np.abs(d1 - vsu2) < 1e-12) | (np.abs(d2 - vsu2) < 1e-12))
assert all((np.abs(d1 - vo2) < 1e-12) | (np.abs(d2 - vo2) < 1e-12))
assert all((np.abs(d1 - vu1) < 1e-12) | (np.abs(d2 - vu1) < 1e-12))
assert all((np.abs(d1 - vas) < 1e-12) | (np.abs(d2 - vas) < 1e-12))


# test implicit matrix
def matmat(x):
    return mat_su2 @ (mat_su2 @ x)


vals = ST_SU2.eigs(
    matmat,
    nvals,
    dmax_full=dmax_full,
    rng=rng,
    reps=mat_su2.row_reps,
    signature=mat_su2.signature[: mat_su2.n_row_reps],
    dtype=mat_su2.dtype,
)
vals = vals.toarray(sort=True)[:nvals]
assert all((np.abs(d1**2 - vals) < 1e-12) | (np.abs(d2**2 - vals) < 1e-12))


# test missing blocks
inds = np.array([1, 2, 4])
blocks = [mat_su2.blocks[i] for i in inds]
block_irreps = mat_su2.block_irreps[inds]
mat_missing = ST_SU2(reps, reps, blocks, block_irreps, mat_su2.signature)
v1 = mat_missing.eigs(mat_missing, 40, dmax_full=200, rng=rng)
v2 = mat_missing.eigs(mat_missing, 4, dmax_full=2, rng=rng)


# implicit matrix with missing blocks
def matmat(x):
    return mat_missing @ (mat_missing @ x)


vals = ST_SU2.eigs(
    matmat,
    40,
    dmax_full=200,
    rng=rng,
    reps=mat_su2.row_reps,
    signature=mat_su2.signature[: mat_su2.n_row_reps],
    dtype=mat_su2.dtype,
)
vals = ST_SU2.eigs(
    matmat,
    4,
    dmax_full=2,
    rng=rng,
    reps=mat_su2.row_reps,
    signature=mat_su2.signature[: mat_su2.n_row_reps],
    dtype=mat_su2.dtype,
)


# pathological case: nblocks = 0
block_irreps = np.array([], dtype=int)
mat_0b = ST_SU2(reps, reps, [], block_irreps, mat_su2.signature)
assert mat_0b.norm() == 0.0
v1 = mat_0b.eigs(mat_0b, 40, dmax_full=200, rng=rng)
v2 = mat_0b.eigs(mat_0b, 4, dmax_full=2, rng=rng)
assert v1.shape == (0,)
assert v2.shape == (0,)


# implicit matrix returning no block
def matmat0(x):
    return mat_0b @ (mat_0b @ x)


vals = ST_SU2.eigs(
    matmat0,
    40,
    dmax_full=200,
    rng=rng,
    reps=mat_su2.row_reps,
    signature=mat_su2.signature[: mat_su2.n_row_reps],
    dtype=mat_su2.dtype,
)
assert vals.shape == (0,)
vals = ST_SU2.eigs(
    matmat0,
    4,
    dmax_full=2,
    rng=rng,
    reps=mat_su2.row_reps,
    signature=mat_su2.signature[: mat_su2.n_row_reps],
    dtype=mat_su2.dtype,
)
assert vals.shape == (0,)


# test return_eigenvectors
vals, vec = mat_su2.eigs(
    mat_su2, nvals, dmax_full=dmax_full, rng=rng, return_eigenvectors=True
)
assert vals.nblocks == vec.nblocks
assert (vals.block_irreps == vec.block_irreps).all()
assert all(
    vec.blocks[i].shape[1] == vals.diagonal_blocks[i].shape[0]
    for i in range(vals.nblocks)
)
assert (vec * vals - mat_su2 @ vec).norm() < 1e-12

# return_eigenvectors with missing blocks
vals, vec = ST_SU2.eigs(
    mat_missing, nvals, dmax_full=dmax_full, rng=rng, return_eigenvectors=True
)
assert vals.nblocks == vec.nblocks
assert (vals.block_irreps == vec.block_irreps).all()
assert all(
    vec.blocks[i].shape[1] == vals.diagonal_blocks[i].shape[0]
    for i in range(vals.nblocks)
)
assert (vec * vals - mat_missing @ vec).norm() < 1e-12


# test eigsh
blocks = [b + b.T.conj() for b in mat_su2.blocks]
block_irreps = mat_su2.block_irreps
mat_sym = ST_SU2(reps, reps, blocks, block_irreps, mat_su2.signature)
vals = ST_SU2.eigsh(mat_sym, nvals, dmax_full=dmax_full, rng=rng)
assert vals.dtype == np.float64

# test eigenvectors
vals, vec = ST_SU2.eigsh(
    mat_sym, nvals, dmax_full=dmax_full, rng=rng, return_eigenvectors=True
)
assert vals.nblocks == vec.nblocks
assert (vals.block_irreps == vec.block_irreps).all()
assert all(
    vec.blocks[i].shape[1] == vals.diagonal_blocks[i].shape[0]
    for i in range(vals.nblocks)
)
assert (vec * vals - mat_sym @ vec).norm() < 1e-12
check = (vec.dagger() @ mat_sym @ vec).blocks
assert all(
    lg.norm(check[i] - np.diag(d)) < 1e-12 for i, d in enumerate(vals.diagonal_blocks)
)

# test eigenvectors for zero block
vals, vec = ST_SU2.eigsh(
    mat_0b, nvals, dmax_full=dmax_full, rng=rng, return_eigenvectors=True
)
assert vals.shape == (0,)
assert vals.nblocks == vec.nblocks == 0
assert (vec * vals - mat_0b @ vec).norm() < 1e-12  # should still be possble


# test dense eig
vals, vec = mat_su2.eig()
assert vals.nblocks == vec.nblocks == mat_su2.nblocks
assert (vals.block_irreps == mat_su2.block_irreps).all()
assert (vals.block_irreps == mat_su2.block_irreps).all()
assert all(b1.shape == b2.shape for b1, b2 in zip(vec.blocks, mat_su2.blocks))
assert all(
    b2.shape == (b2.shape[0],) for b1, b2 in zip(vec.blocks, vals.diagonal_blocks)
)
assert (vec * vals - mat_su2 @ vec).norm() < 1e-12

# test dense eigh
vals, vec = mat_sym.eigh()
assert vals.nblocks == vec.nblocks == mat_sym.nblocks
assert (vals.block_irreps == mat_sym.block_irreps).all()
assert (vals.block_irreps == mat_sym.block_irreps).all()
assert all(b1.shape == b2.shape for b1, b2 in zip(vec.blocks, mat_sym.blocks))
assert all(
    b2.shape == (b2.shape[0],) for b1, b2 in zip(vec.blocks, vals.diagonal_blocks)
)
assert (vec * vals - mat_sym @ vec).norm() < 1e-12
assert (vec * vals @ vec.dagger() - mat_sym).norm() < 1e-12
