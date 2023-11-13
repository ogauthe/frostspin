#!/usr/bin/env python3

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.su2_symmetric_tensor import SU2_SymmetricTensor

sds_22 = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)

sds_22t = sds_22.reshape(2, 2, 2, 2)
r = np.array([[1], [2]])

st = SU2_SymmetricTensor.from_array(sds_22t, (r, r), (r, r))
assert st.nblocks == 2
assert (st.block_irreps == np.array([1, 3])).all()
assert st.blocks[0].shape == (1, 1)
assert st.blocks[1].shape == (1, 1)
assert abs(st.blocks[0] + 0.75) < 1e-14
assert abs(st.blocks[1] - 0.25) < 1e-14
assert abs(lg.norm(sds_22) - st.norm()) < 1e-14
assert lg.norm(st.toarray() - sds_22t) < 1e-14

stp = st.permutate((0, 1, 2), (3,))
stp2 = SU2_SymmetricTensor.from_array(sds_22t, (r, r, r), (r,), signature=[0, 0, 1, 1])
assert (stp - stp2).norm() < 1e-14
st2 = stp.permutate((0, 1), (2, 3))
assert (st - st2).norm() < 1e-14

stp = st.permutate((3, 0, 2), (1,))
arrp = sds_22t.transpose((3, 0, 2, 1))
stp2 = SU2_SymmetricTensor.from_array(arrp, (r, r, r), (r,), signature=[1, 0, 1, 0])
assert (stp - stp2).norm() < 1e-14
st2 = stp.permutate((1, 3), (2, 0))
assert (st - st2).norm() < 1e-14

sds_22b = np.array(
    [
        [-0.25, 0.0, 0.0, -0.5],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0],
        [-0.5, 0.0, 0.0, -0.25],
    ]
)
arrb = sds_22b.reshape(2, 2, 2, 2)
stb = SU2_SymmetricTensor.from_array(arrb, (r, r), (r, r), signature=[0, 1, 1, 0])
assert stb.nblocks == 2
assert (stb.block_irreps == np.array([1, 3])).all()
assert stb.blocks[0].shape == (1, 1)
assert stb.blocks[1].shape == (1, 1)
assert abs(stb.blocks[0] + 0.75) < 1e-14
assert abs(stb.blocks[1] - 0.25) < 1e-14
assert abs(lg.norm(sds_22) - stb.norm()) < 1e-14
assert lg.norm(stb.toarray() - arrb) < 1e-14

stbp = stb.permutate((0, 1, 2), (3,))
stbp2 = SU2_SymmetricTensor.from_array(arrb, (r, r, r), (r,), signature=[0, 1, 1, 0])
assert (stbp - stbp2).norm() < 1e-14
stb2 = stbp.permutate((0, 1), (2, 3))
assert (stb - stb2).norm() < 1e-14

stbp = stb.permutate((3, 0, 2), (1,))
arrbp = arrb.transpose((3, 0, 2, 1))
stbp2 = SU2_SymmetricTensor.from_array(arrbp, (r, r, r), (r,), signature=[0, 0, 1, 1])
assert (stbp - stbp2).norm() < 1e-14
stb2 = stbp.permutate((1, 3), (2, 0))
assert (stb - stb2).norm() < 1e-14


# try tensor with external degeneracies
r1 = np.array([[1], [1]])
r12 = np.array([[1, 2, 1], [1, 3, 5]])
arr = np.eye(12).reshape(1, 12, 12, 1)
st3 = SU2_SymmetricTensor.from_array(arr, (r1,), (r12, r12, r1), signature=[0, 0, 1, 1])
assert lg.norm(st3.toarray() - arr) < 1e-14


# check cast
stu1 = st.toU1()
assert lg.norm(stu1.toarray() - sds_22t) < 1e-14
sto2 = st.toO2()
assert lg.norm(sto2.toarray() - sds_22t) < 1e-14

# check merge_legs
_ = st.merge_legs(0, 1)
_ = st.merge_legs(2, 3)


# check missing blocks
r3 = np.array([[1], [3]])
blocks = (-np.eye(1), 2 * np.eye(1))
block_irreps = np.array([1, 5])
sign = np.array([False, False, True, True])
t = SU2_SymmetricTensor.random((r3, r3, r3), (r3,))
t = t @ t.H  # there is only one block among the many allowed
tp = t.permutate((3, 2, 1), (0, 5, 4))
tp2 = tp.permutate((3, 2, 1), (0, 5, 4))
assert (tp2 - t).norm() < 1e-14


# try I/O for isometries
st.save_isometries("data_test_su2_isometries.npz")
st.load_isometries("data_test_su2_isometries.npz")
