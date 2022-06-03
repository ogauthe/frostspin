#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor

SdS_22 = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)

arr = SdS_22.reshape(2, 2, 2, 2)
r = np.array([[1], [2]])
nf = lg.norm(arr)
i2 = np.array([[1], [2]])

st = SU2_SymmetricTensor.from_array(arr, (r, r), (r, r))
assert st.nblocks == 2
assert (st.block_irreps == np.array([1, 3])).all()
assert st.blocks[0].shape == (1, 1)
assert st.blocks[1].shape == (1, 1)
assert abs(st.blocks[0] + 0.75) < 1e-14
assert abs(st.blocks[1] - 0.25) < 1e-14
assert abs(1.0 - st.norm() / nf) < 1e-14
assert lg.norm(st.toarray() - arr) / nf < 1e-14

stp = st.permutate((0, 1, 2), (3,))

SdS_22b = np.array(
    [
        [-0.25, 0.0, 0.0, -0.5],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0],
        [-0.5, 0.0, 0.0, -0.25],
    ]
)
arrb = SdS_22b.reshape(2, 2, 2, 2)
stb = SU2_SymmetricTensor.from_array(arrb, (r, r), (r, r), signature=[0, 1, 1, 0])
assert stb.nblocks == 2
assert (stb.block_irreps == np.array([1, 3])).all()
assert stb.blocks[0].shape == (1, 1)
assert stb.blocks[1].shape == (1, 1)
assert abs(stb.blocks[0] + 0.75) < 1e-14
assert abs(stb.blocks[1] - 0.25) < 1e-14
assert abs(1.0 - stb.norm() / nf) < 1e-14
assert lg.norm(stb.toarray() - arrb) / nf < 1e-14

stbp = stb.permutate((0, 1, 2), (3,))
