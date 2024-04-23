#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from froSTspin.symmetric_tensor.o2_symmetric_tensor import O2_SymmetricTensor
from froSTspin.symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor

rng = np.random.default_rng(42)
r1 = np.array([[1, 1, 1, 1], [-1, 0, 1, 2]])
r2 = np.array([[2, 1, 1, 1], [-1, 0, 1, 2]])
r3 = np.array([[1, 2, 1, 1], [-1, 0, 1, 2]])
r4 = np.array([[1, 1, 1, 1], [-1, 0, 2, 3]])
r5 = np.array([[2, 1, 1, 1], [-1, 0, 2, 3]])
row_reps = (r1, r2, r3)
col_reps = (r4, r5)


def check_o2(to2, tu1):
    assert lg.norm(tu1.toarray() - to2.toarray()) < 1e-15
    temp = O2_SymmetricTensor.from_U1(tu1, to2.row_reps, to2.col_reps)
    assert (temp - to2).norm() < 1e-15
    row_axes, col_axes = (2, 1, 4), (0, 3)
    tpo2 = to2.permute(row_axes, col_axes)
    tpu1 = tu1.permute(row_axes, col_axes)

    assert (tpo2.toU1() - tpu1).norm() < 1e-15
    assert lg.norm(tpo2.toarray() - tpu1.toarray()) < 1e-15

    tpo2 = to2.transpose()
    tpu1 = tu1.transpose()
    assert (tpo2.toU1() - tpu1).norm() < 1e-15
    assert lg.norm(tpo2.toarray() - tpu1.toarray()) < 1e-15


assert O2_SymmetricTensor.symmetry() == "O2"
assert O2_SymmetricTensor.singlet().shape == (2, 1)
assert (O2_SymmetricTensor.singlet() == np.array([[1], [0]])).all()

to2 = O2_SymmetricTensor.random(row_reps, col_reps, rng=rng)
to2 /= to2.norm()
tu1 = to2.toU1()

check_o2(to2, tu1)


# Pathological cases: missing 0e/0o/other blocks
t2o2 = to2.copy()
t2o2.blocks[0][:] = 0
t2u1 = t2o2.toU1()
t2o2 = O2_SymmetricTensor(  # remove 0odd block
    t2o2.row_reps,
    t2o2.col_reps,
    t2o2.blocks[1:],
    t2o2.block_irreps[1:],
    t2o2.signature,
)
check_o2(t2o2, t2u1)

t2o2 = to2.copy()
t2o2.blocks[1][:] = 0
t2u1 = t2o2.toU1()
t2o2 = O2_SymmetricTensor(  # remove 0even block
    t2o2.row_reps,
    t2o2.col_reps,
    t2o2.blocks[:1] + t2o2.blocks[2:],
    (-1, *t2o2.block_irreps[2:]),
    to2.signature,
)
check_o2(t2o2, t2u1)

t2o2 = to2.copy()
t2o2.blocks[0][:] = 0
t2o2.blocks[1][:] = 0
t2u1 = t2o2.toU1()
t2o2 = O2_SymmetricTensor(  # remove both 0even and 0odd
    t2o2.row_reps, t2o2.col_reps, t2o2.blocks[2:], t2o2.block_irreps[2:], t2o2.signature
)
check_o2(t2o2, t2u1)

t2o2 = to2.copy()
t2o2 = O2_SymmetricTensor(  # remove everything but 0o/0e
    t2o2.row_reps, t2o2.col_reps, t2o2.blocks[:2], t2o2.block_irreps[:2], t2o2.signature
)
t2u1 = U1_SymmetricTensor(  # remove everything but 0o/0e
    tu1.row_reps, tu1.col_reps, (tu1.blocks[6],), (0,), tu1.signature
)
check_o2(t2o2, t2u1)


r6 = np.array([[1, 1, 1], [2, 3, 4]])
r7 = np.array([[2, 1], [2, 3]])
to2 = O2_SymmetricTensor.random((r1, r2, r6), (r3, r7), rng=rng)
to2 /= to2.norm()
tu1 = to2.toU1()
to2 = to2.permute((0, 1, 3), (2, 4))  # no fixed points in columns, only doublets
tu1 = tu1.permute((0, 1, 3), (2, 4))
check_o2(to2, tu1)


r6 = np.array([[4], [-1]])
r7 = np.array([[5], [-1]])
to2 = O2_SymmetricTensor.random((r1, r2, r6), (r3, r7), rng=rng)
to2 /= to2.norm()
tu1 = to2.toU1()
to2 = to2.permute((0, 1, 3), (2, 4))  # only odd fixed points, no even or doublet
tu1 = tu1.permute((0, 1, 3), (2, 4))
check_o2(to2, tu1)

r6 = np.array([[4], [0]])
r7 = np.array([[5], [0]])
to2 = O2_SymmetricTensor.random((r1, r2, r6), (r3, r7), rng=rng)
to2 /= to2.norm()
tu1 = to2.toU1()
to2 = to2.permute((0, 1, 3), (2, 4))  # only even fixed points, no odd or doublet
tu1 = tu1.permute((0, 1, 3), (2, 4))
check_o2(to2, tu1)
