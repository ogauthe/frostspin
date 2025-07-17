#!/usr/bin/env python

import numpy as np
import scipy.linalg as lg

from frostspin import SU2SymmetricTensor

rng = np.random.default_rng(42)
reps = (np.array([[2, 3, 1], [1, 3, 5]]), np.array([[2, 2, 1], [1, 3, 5]]))
mat_su2 = SU2SymmetricTensor.random(reps, reps, rng=rng)
mat_su2 /= mat_su2.norm()
mat_o2 = mat_su2.toO2()
mat_u1 = mat_su2.toU1()
mat_as = mat_su2.totrivial()
dense = mat_su2.toarray(as_matrix=True)

default = lg.eigvals(dense)
default = default[np.abs(default).argsort()[::-1]]


nvals = 20
dmax_full = 10


vsu2 = mat_su2.truncated_eig(nvals, dmax_full=dmax_full, rng=rng, compute_vectors=False)
assert vsu2.dtype == np.complex128

vsu2 = vsu2.toarray(sort=True)[:nvals]
vo2 = mat_o2.truncated_eig(nvals, dmax_full=dmax_full, rng=rng, compute_vectors=False)
vo2 = vo2.toarray(sort=True)[:nvals]
vu1 = mat_u1.truncated_eig(nvals, dmax_full=dmax_full, rng=rng, compute_vectors=False)
vu1 = vu1.toarray(sort=True)[:nvals]
vas = mat_as.truncated_eig(nvals, dmax_full=dmax_full, rng=rng, compute_vectors=False)
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


vals = SU2SymmetricTensor.eigs(
    matmat,
    nvals,
    mat_su2.row_reps,
    mat_su2.signature[: mat_su2.n_row_reps],
    dmax_full=dmax_full,
    rng=rng,
    compute_vectors=False,
)
vals = vals.toarray(sort=True)[:nvals]
assert all((np.abs(d1**2 - vals) < 1e-12) | (np.abs(d2**2 - vals) < 1e-12))


# test missing blocks
inds = np.array([1, 2, 4])
blocks = [mat_su2.blocks[i] for i in inds]
block_irreps = mat_su2.block_irreps[inds]
mat_missing = SU2SymmetricTensor(reps, reps, blocks, block_irreps, mat_su2.signature)
v1 = mat_missing.truncated_eig(40, dmax_full=200, rng=rng, compute_vectors=False)
v2 = mat_missing.truncated_eig(4, dmax_full=2, rng=rng, compute_vectors=False)


# implicit matrix with missing blocks
def matmat(x):
    return mat_missing @ (mat_missing @ x)


vals = SU2SymmetricTensor.eigs(
    matmat,
    40,
    mat_su2.row_reps,
    mat_su2.signature[: mat_su2.n_row_reps],
    dmax_full=200,
    rng=rng,
    compute_vectors=False,
)
vals = SU2SymmetricTensor.eigs(
    matmat,
    4,
    mat_su2.row_reps,
    mat_su2.signature[: mat_su2.n_row_reps],
    dmax_full=2,
    rng=rng,
    compute_vectors=False,
)


# pathological case: nblocks = 0
block_irreps = np.array([], dtype=int)
mat_0b = SU2SymmetricTensor(reps, reps, [], block_irreps, mat_su2.signature)
assert mat_0b.norm() == 0.0
v1 = mat_0b.truncated_eig(40, dmax_full=200, rng=rng, compute_vectors=False)
v2 = mat_0b.truncated_eig(4, dmax_full=2, rng=rng, compute_vectors=False)
assert v1.shape == (0,)
assert v2.shape == (0,)


# implicit matrix returning no block
def matmat0(x):
    return mat_0b @ (mat_0b @ x)


vals = SU2SymmetricTensor.eigs(
    matmat0,
    40,
    mat_su2.row_reps,
    mat_su2.signature[: mat_su2.n_row_reps],
    dmax_full=200,
    rng=rng,
    compute_vectors=False,
)
assert vals.shape == (0,)
vals = SU2SymmetricTensor.eigs(
    matmat0,
    4,
    mat_su2.row_reps,
    mat_su2.signature[: mat_su2.n_row_reps],
    dmax_full=2,
    rng=rng,
    compute_vectors=False,
)
assert vals.shape == (0,)


# test computing eigenvectors
vals, vec = mat_su2.truncated_eig(nvals, dmax_full=dmax_full, rng=rng)
assert vals.nblocks == vec.nblocks
assert (vals.block_irreps == vec.block_irreps).all()
assert all(
    vec.blocks[i].shape[1] == vals.diagonal_blocks[i].shape[0]
    for i in range(vals.nblocks)
)
assert (vec * vals - mat_su2 @ vec).norm() < 1e-12

# compute eigenvectors with missing blocks
vals, vec = mat_missing.truncated_eig(nvals, dmax_full=dmax_full, rng=rng)
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
mat_sym = SU2SymmetricTensor(reps, reps, blocks, block_irreps, mat_su2.signature)
vals = mat_sym.truncated_eigh(
    nvals, dmax_full=dmax_full, rng=rng, compute_vectors=False
)
assert vals.dtype == np.float64


def matmat(v):
    return mat_sym @ v


vals = SU2SymmetricTensor.eigsh(
    matmat,
    nvals,
    mat_sym.row_reps,
    mat_sym.signature[:2],
    dmax_full=dmax_full,
    rng=rng,
    compute_vectors=False,
)
assert vals.dtype == np.float64


# test eigenvectors
vals, vec = mat_sym.truncated_eigh(nvals, dmax_full=dmax_full, rng=rng)
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
vals, vec = mat_0b.truncated_eigh(nvals, dmax_full=dmax_full, rng=rng)
assert vals.shape == (0,)
assert vals.nblocks == vec.nblocks == 0
assert (vec * vals - mat_0b @ vec).norm() < 1e-12  # should still be possble


# test dense eig
vals, vec = mat_su2.eig()
assert vals.nblocks == vec.nblocks == mat_su2.nblocks
assert (vals.block_irreps == mat_su2.block_irreps).all()
assert (vals.block_irreps == mat_su2.block_irreps).all()
assert all(
    b1.shape == b2.shape for b1, b2 in zip(vec.blocks, mat_su2.blocks, strict=True)
)
assert all(
    b2.shape == (b2.shape[0],)
    for b1, b2 in zip(vec.blocks, vals.diagonal_blocks, strict=True)
)
assert (vec * vals - mat_su2 @ vec).norm() < 1e-12

# test dense eigh
vals, vec = mat_sym.eigh()
assert vals.nblocks == vec.nblocks == mat_sym.nblocks
assert (vals.block_irreps == mat_sym.block_irreps).all()
assert (vals.block_irreps == mat_sym.block_irreps).all()
assert all(
    b1.shape == b2.shape for b1, b2 in zip(vec.blocks, mat_sym.blocks, strict=True)
)
assert all(
    b2.shape == (b2.shape[0],)
    for b1, b2 in zip(vec.blocks, vals.diagonal_blocks, strict=True)
)
assert (vec * vals - mat_sym @ vec).norm() < 1e-12
assert (vec * vals @ vec.dagger() - mat_sym).norm() < 1e-12
