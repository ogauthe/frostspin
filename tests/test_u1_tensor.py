#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from frostspin import U1SymmetricTensor


def random_U1_tensor(row_reps, col_reps, signature, rng=None):
    """
    Construct random U(1) symmetric tensor. Non-zero coefficients are taken from
    continuous uniform distribution in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    row_reps : tuple of 1D integer arrays.
        U(1) quantum numbers of row axes.
    col_reps : tuple of 1D integer arrays.
        U(1) quantum numbers of column axes.
    signature : 1D array with bool dtype
        signature of each representation in row_reps + col_reps
    rng : optional, random number generator. Can be used to reproduce results.

    Returns
    -------
    output : ndarray
        random U(1) tensor, with shape following axis_reps dimensons.
    """
    if rng is None:
        rng = np.random.default_rng()
    reps = row_reps + col_reps
    irreps1D = U1SymmetricTensor.combine_representations(reps, signature)
    nnz = (irreps1D == 0).nonzero()[0]
    t0 = np.zeros(irreps1D.size)
    t0[nnz] = rng.random(nnz.size) - 0.5
    return t0.reshape(tuple(r.size for r in reps)) / lg.norm(t0)


rng = np.random.default_rng(42)

c1 = np.array([-2, 0, 2], dtype=np.int8)
c2 = np.array([-3, -1, 1, 3], dtype=np.int8)
c3 = np.array([-2, -2, 0, 0, 0, 2, 2], dtype=np.int8)
c4 = np.array([-2, -1, 0, 1, 2], dtype=np.int8)
c5 = np.array([-3, -1, 0, 2], dtype=np.int8)  # breaks conjugation symmetry

row_reps = (c1, c2)
col_reps = (c3, c4, c5)
signature = np.array([1, 0, 0, 1, 1], dtype=bool)

assert U1SymmetricTensor.symmetry() == "U1"
assert U1SymmetricTensor.singlet().shape == (1,)
assert (U1SymmetricTensor.singlet() == np.zeros((1,), dtype=np.int8)).all()


t0 = random_U1_tensor(row_reps, col_reps, signature=signature, rng=rng)
tu1 = U1SymmetricTensor.from_array(t0, row_reps, col_reps, signature=signature)

assert (tu1.toarray() == t0).all()
assert abs(1.0 - tu1.norm() / lg.norm(t0)) < 1e-14
assert (tu1.transpose().toarray() == t0.transpose(2, 3, 4, 0, 1)).all()
assert tu1.permute((0, 1), (2, 3, 4)) is tu1

temp = tu1.permute((2, 3, 4), (0, 1))
assert temp.nblocks == tu1.nblocks
assert all(
    (b1 == b2).all() for b1, b2 in zip(temp.blocks, tu1.transpose().blocks, strict=True)
)
temp = tu1.permute((3, 0, 2), (1, 4))
assert (temp.toarray() == t0.transpose(3, 0, 2, 1, 4)).all()
temp = tu1.permute((3,), (1, 0, 4, 2))
assert (temp.toarray() == t0.transpose(3, 1, 0, 4, 2)).all()

temp = tu1.dual()
assert (temp.toarray() == t0).all()
temp = temp.permute((3, 0, 2), (1, 4))
assert (temp.toarray() == t0.transpose(3, 0, 2, 1, 4)).all()

u, s, v = tu1.svd()
s2 = s.copy()
us = u * s2
assert (tu1 - us @ v).norm() < 1e-12
sv = s * v
assert (tu1 - u @ sv).norm() < 1e-12
s12 = s**0.5
us2 = u * s12
s2v = s12 * v
assert (tu1 - us2 @ s2v).norm() < 1e-12
sigma = type(tu1).from_diagonal_tensor(s)
assert (tu1 - u @ sigma @ v).norm() < 1e-12

tu1 = U1SymmetricTensor.random(row_reps, col_reps, signature=signature, rng=rng)
assert abs(tu1.norm() ** 2 - tu1.full_contract(tu1.dagger())) < 1e-12 * tu1.norm() ** 2
