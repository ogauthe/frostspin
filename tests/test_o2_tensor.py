#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.o2_symmetric_tensor import O2_SymmetricTensor

r1 = np.array([[1, 1, 1], [-1, 0, 2]])
r2 = np.array([[2, 1, 1], [-1, 0, 2]])
r3 = np.array([[1, 2, 1], [-1, 0, 2]])
r4 = np.array([[1, 1, 1, 1], [-1, 0, 2, 4]])
r5 = np.array([[2, 1, 1, 1], [-1, 0, 2, 4]])


row_reps = (r1, r2, r3)
col_reps = (r4, r5)
to2 = O2_SymmetricTensor.random(row_reps, col_reps)
to2 /= to2.norm()

tu1 = to2.toU1()
assert lg.norm(tu1.toarray() - to2.toarray()) < 1e-15
assert (O2_SymmetricTensor.from_U1(tu1, row_reps, col_reps) - to2).norm() < 1e-15

row_axes, col_axes = (2, 1, 4), (0, 3)
tpo2 = to2.permutate(row_axes, col_axes)
tpu1 = tu1.permutate(row_axes, col_axes)

assert (tpo2.toU1() - tpu1).norm() < 1e-15
assert lg.norm(tpo2.toarray() - tpu1.toarray()) < 1e-15
