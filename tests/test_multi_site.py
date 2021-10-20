#!/usr/bin/env python3

import numpy as np

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor
from ctmrg.ctmrg import CTMRG_U1

ap = np.array([-1, 1], dtype=np.int8)
aa = np.array([-1, 1], dtype=np.int8)
a1 = np.array([-1, 0, 1], dtype=np.int8)
a2 = np.array([-2, 0, 2], dtype=np.int8)
a3 = np.array([-1, 1, 0], dtype=np.int8)
a4 = np.array([-2, 2, 0], dtype=np.int8)

axes1 = np.array(
    [
        [0, 0, -1, 1],
        [0, 0, -2, 2],
        [0, 1, -1, 0],
        [0, 1, 0, -1],
        [0, 2, -2, 0],
        [0, 2, 0, -2],
        [-1, 0, 0, 1],
        [-1, 0, 1, 0],
        [-1, 1, 0, 0],
        [-1, -1, 1, 1],
        [-1, 1, -1, 1],
        [-1, 1, -2, 2],
        [-1, 1, 2, -2],
        [-1, -2, 2, 1],
        [-1, 2, -2, 1],
        [-1, -2, 1, 2],
        [-1, 2, 1, -2],
        [-2, 0, 0, 2],
        [-2, 0, 2, 0],
        [-2, -1, 1, 2],
        [-2, 1, -1, 2],
        [-2, -1, 2, 1],
        [-2, 1, 2, -1],
        [-2, 2, 0, 0],
        [-2, -2, 2, 2],
        [-2, 2, -2, 2],
        [-2, 2, -1, 1],
        [-2, 2, 1, -1],
        [-3, 0, 0, 3],
        [-3, 0, 3, 0],
        [-3, -1, 1, 3],
        [-3, 1, -1, 3],
        [-3, -1, 3, 1],
        [-3, 1, 3, -1],
        [-3, 3, 0, 0],
        [-3, -3, 3, 3],
        [-3, 3, -3, 3],
        [-3, 3, -1, 1],
        [-3, 3, 1, -1],
    ],
    dtype=np.int8,
)

nx = axes1.shape[0]
d = axes1.shape[1]

# check no doublet in axes1
print(((axes1 == axes1[:, None]).all(axis=2) == np.eye(nx, dtype=bool)).all())
# check no doublet up to sign
print((axes1 == -axes1[:, None]).all(axis=2).sum() == 0)

# bilayer with sign swap
axes2 = (axes1[:, :, None] - axes1[:, None]).reshape(nx, d ** 2)
print(((axes2 == axes2[:, None]).all(axis=2) == np.eye(nx, dtype=bool)).all())
print((axes2 == -axes2[:, None]).all(axis=2).sum() == 0)

representations = (
    (ap, aa, axes1[0], axes1[1], axes1[2], axes1[3]),
    (-ap, -aa, -axes1[4], -axes1[5], -axes1[6], -axes1[1]),
    (ap, aa, axes1[7], axes1[8], axes1[9], axes1[5]),
    (-ap, -aa, -axes1[10], -axes1[3], -axes1[11], -axes1[8]),
    (-ap, -aa, -axes1[2], -axes1[12], -axes1[13], -axes1[14]),
    (ap, aa, axes1[6], axes1[15], axes1[16], axes1[12]),
    (-ap, -aa, -axes1[9], -axes1[17], -axes1[18], -axes1[15]),
    (ap, aa, axes1[11], axes1[14], axes1[19], axes1[17]),
    (ap, aa, axes1[13], axes1[20], axes1[21], axes1[22]),
    (-ap, -aa, -axes1[16], -axes1[23], -axes1[24], -axes1[20]),
    (ap, aa, axes1[18], axes1[25], axes1[26], axes1[23]),
    (-ap, -aa, -axes1[19], -axes1[22], -axes1[27], -axes1[25]),
    (-ap, -aa, -axes1[21], -axes1[28], -axes1[0], -axes1[29]),
    (ap, aa, axes1[24], axes1[30], axes1[4], axes1[28]),
    (-ap, -aa, -axes1[26], -axes1[31], -axes1[7], -axes1[30]),
    (ap, aa, axes1[27], axes1[29], axes1[10], axes1[31]),
)

rng = np.random.default_rng(42)
t00 = U1_SymmetricTensor.random(representations[0], 2, rng=rng).toarray()
t01 = U1_SymmetricTensor.random(representations[1], 2, rng=rng).toarray()
t02 = U1_SymmetricTensor.random(representations[2], 2, rng=rng).toarray()
t03 = U1_SymmetricTensor.random(representations[3], 2, rng=rng).toarray()

t10 = U1_SymmetricTensor.random(representations[4], 2, rng=rng).toarray()
t11 = U1_SymmetricTensor.random(representations[5], 2, rng=rng).toarray()
t12 = U1_SymmetricTensor.random(representations[6], 2, rng=rng).toarray()
t13 = U1_SymmetricTensor.random(representations[7], 2, rng=rng).toarray()

t20 = U1_SymmetricTensor.random(representations[8], 2, rng=rng).toarray()
t21 = U1_SymmetricTensor.random(representations[9], 2, rng=rng).toarray()
t22 = U1_SymmetricTensor.random(representations[10], 2, rng=rng).toarray()
t23 = U1_SymmetricTensor.random(representations[11], 2, rng=rng).toarray()

t30 = U1_SymmetricTensor.random(representations[12], 2, rng=rng).toarray()
t31 = U1_SymmetricTensor.random(representations[13], 2, rng=rng).toarray()
t32 = U1_SymmetricTensor.random(representations[14], 2, rng=rng).toarray()
t33 = U1_SymmetricTensor.random(representations[15], 2, rng=rng).toarray()

tensors = (
    t00,
    t01,
    t02,
    t03,
    t10,
    t11,
    t12,
    t13,
    t20,
    t21,
    t22,
    t23,
    t30,
    t31,
    t32,
    t33,
)
tiling = "ABCD\nEFGH\nIJKL\nMNOP"
ctm = CTMRG_U1.from_elementary_tensors(
    tiling,
    tensors,
    representations,
    21,
    block_chi_ratio=1.2,
    cutoff=1e-10,
    degen_ratio=1.0,
    verbosity=2,
)

ctm.iterate()
ctm.iterate()
