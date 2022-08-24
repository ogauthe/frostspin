#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.o2_symmetric_tensor import O2_SymmetricTensor
from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor

r1 = np.array([[1, 1, 1, 1], [-1, 0, 1, 2]])
r2 = np.array([[2, 1, 1, 1], [-1, 0, 1, 2]])
r3 = np.array([[1, 2, 1, 1], [-1, 0, 2, 2]])
r4 = np.array([[1, 1, 1, 1], [-1, 0, 2, 3]])
r5 = np.array([[2, 1, 1, 1], [-1, 0, 2, 3]])


row_reps = (r1, r2, r3)
col_reps = (r4, r5)
rng = np.random.default_rng(42)
to2 = O2_SymmetricTensor.random(row_reps, col_reps, rng=rng)
to2 /= to2.norm()

tu1 = to2.toU1()
assert lg.norm(tu1.toarray() - to2.toarray()) < 1e-15
assert (O2_SymmetricTensor.from_U1(tu1, row_reps, col_reps) - to2).norm() < 1e-15

row_axes, col_axes = (2, 1, 4), (0, 3)
tpo2 = to2.permutate(row_axes, col_axes)
tpu1 = tu1.permutate(row_axes, col_axes)

assert (tpo2.toU1() - tpu1).norm() < 1e-15
assert lg.norm(tpo2.toarray() - tpu1.toarray()) < 1e-15

tpo2 = to2.T
tpu1 = tu1.T
assert (tpo2.toU1() - tpu1).norm() < 1e-15
assert lg.norm(tpo2.toarray() - tpu1.toarray()) < 1e-15

th = to2.H
th2 = tpo2.conjugate()
th3 = to2.conjugate().T
assert (th - th2).norm() < 1e-15
assert (th - th3).norm() < 1e-15

# SU2.toO2() adds some swaps and some signs, cannot compare toarray()
# just test whether calling it crashes
rrsu2 = (np.array([[2, 2, 2, 2, 2, 2], [1, 2, 3, 4, 5, 6]]),)
rcsu2 = (np.array([[1, 1, 1], [1, 2, 3]]), np.array([[2, 1, 1], [1, 3, 5]]))
tsu2 = SU2_SymmetricTensor.random(rrsu2, rcsu2, rng=rng)
to2 = tsu2.toO2()
