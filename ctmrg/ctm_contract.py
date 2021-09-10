import numpy as np
import numba

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor

###############################################################################
#  construct 2x2 corners
#  memory: peak at 2*a*d*chi**2*D**4
###############################################################################


def contract_corner(C1, T1, T4, A):
    """
    Contract a corner C-T//T-A. Take upper left corner as template.
    """
    #  C1-03-T1-2
    #  |     ||
    #  1->3  01
    ul = T1.transpose(1, 2, 0, 3).reshape(T1.shape[1] ** 2 * T1.shape[0], T1.shape[3])
    ul = ul @ C1

    #  C1---T1-2
    #  |    ||
    #  3    01
    #  0
    #  |
    #  T4=1,2 -> 3,4
    #  |
    #  3 -> 5
    ul = (ul @ T4.reshape(T4.shape[0], T4.shape[1] ** 2 * T4.shape[3])).reshape(
        T1.shape[1], T1.shape[2], T1.shape[0], T4.shape[1], T4.shape[2], T4.shape[3]
    )
    ul = (
        ul.transpose(0, 3, 1, 4, 2, 5)
        .copy()
        .reshape(
            T1.shape[1] * T4.shape[1],
            T1.shape[2] * T4.shape[2] * T1.shape[0] * T4.shape[3],
        )
    )
    temp = A.transpose(0, 1, 3, 4, 2, 5).reshape(
        A.shape[0] * A.shape[1] * A.shape[3] * A.shape[4], A.shape[2] * A.shape[5]
    )

    #  C1----T1-4
    #  |     ||
    #  |     02
    #  |   0 4
    #  |    \|
    #  T4-15-A-2
    #  | \3  |\
    #  5     3 1
    ul = (temp @ ul).reshape(
        A.shape[0] * A.shape[1],
        A.shape[3] * A.shape[4],
        T1.shape[2] * T4.shape[2],
        T1.shape[0] * T4.shape[3],
    )
    ul = (
        ul.swapaxes(1, 2)
        .copy()
        .reshape(
            A.shape[0] * A.shape[1] * T1.shape[2] * T4.shape[2],
            A.shape[3] * A.shape[4] * T1.shape[0] * T4.shape[3],
        )
    )  # memory peak 2*a*d*chi**2*D**4
    #  C1----T1-6
    #  |     ||
    #  |     |2
    #  |   0 |        2 4
    #  |    \|         \|
    #  T4----A-4      5-A*-0
    #  | \3  |\         |\
    #  7     5 1        1 3
    temp = (
        temp.reshape(
            A.shape[0] * A.shape[1], A.shape[3] * A.shape[4], A.shape[2] * A.shape[5]
        )
        .swapaxes(0, 1)
        .conj()
        .copy()
        .reshape(
            A.shape[3] * A.shape[4], A.shape[0] * A.shape[1] * A.shape[2] * A.shape[5]
        )
    )
    ul = (temp @ ul).reshape(
        A.shape[3], A.shape[4], A.shape[3], A.shape[4], T1.shape[0], T4.shape[3]
    )

    #  C1-T1-4 ---->2
    #  |  ||
    #  T4=AA*=2,0->0,1
    #  |  ||
    #  5  31
    #  5  34
    return ul.transpose(2, 0, 4, 3, 1, 5)  # do not reshape to avoid copy here


def contract_ul_corner(C1, T1, T4, A):
    return contract_corner(C1, T1, T4, A)


def contract_ur_corner(T1, C2, A, T2):
    return contract_corner(
        C2, T2.transpose(1, 2, 3, 0), T1, A.transpose(0, 1, 3, 4, 5, 2)
    )


def contract_dr_corner(A, T2, T3, C3):
    return contract_corner(
        C3.T,
        T3.transpose(3, 0, 1, 2),
        T2.transpose(1, 2, 3, 0),
        A.transpose(0, 1, 4, 5, 2, 3),
    ).transpose(
        3, 4, 5, 0, 1, 2
    )  # transpose matrix to keep clockwise legs


def contract_dl_corner(T4, A, C4, T3):
    return contract_corner(
        C4, T4, T3.transpose(3, 0, 1, 2), A.transpose(0, 1, 5, 2, 3, 4)
    )


###############################################################################
# construct halves from corners
# memory: max during corner construction
###############################################################################


def contract_u_half(C1, T1l, T1r, C2, T4, Al, Ar, T2):
    ul = contract_ul_corner(C1, T1l, T4, Al).copy()
    ul = ul.reshape(Al.shape[3] ** 2 * T1l.shape[0], Al.shape[4] ** 2 * T4.shape[3])
    ur = contract_ur_corner(T1r, C2, Ar, T2).copy()
    ur = ur.reshape(Ar.shape[4] ** 2 * T2.shape[1], Ar.shape[5] ** 2 * T1r.shape[3])
    #  UL-01-UR
    #  |      |
    #  1      0
    return ur @ ul


def contract_l_half(C1, T1, T4u, Au, T4d, Ad, C4, T3):
    ul = contract_ul_corner(C1, T1, T4u, Au).copy()
    ul = ul.reshape(Au.shape[3] ** 2 * T1.shape[0], Au.shape[4] ** 2 * T4u.shape[3])
    dl = contract_dl_corner(T4d, Ad, C4, T3).copy()
    dl = dl.reshape(Ad.shape[2] ** 2 * T4d.shape[0], Ad.shape[3] ** 2 * T3.shape[2])
    #  UL-0
    #  |
    #  1
    #  0
    #  |
    #  DL-1
    return ul @ dl


def contract_d_half(T4, Al, Ar, T2, C4, T3l, T3r, C3):
    dl = contract_dl_corner(T4, Al, C4, T3l).copy()
    dl = dl.reshape(Al.shape[2] ** 2 * T4.shape[0], Al.shape[3] ** 2 * T3l.shape[2])
    # dr.T is needed in matrix product. Transpose *before* reshape to optimize copy
    dr = contract_dr_corner(Ar, T2, T3r, C3).transpose(3, 4, 5, 0, 1, 2).copy()
    dr = dr.reshape(Ar.shape[5] ** 2 * T3r.shape[3], Ar.shape[2] ** 2 * T2.shape[0])
    #  0      1
    #  0      0
    #  |      |
    #  DL-11-DR
    return dl @ dr


def contract_r_half(T1, C2, Au, T2u, Ad, T2d, T3, C3):
    ur = contract_ur_corner(T1, C2, Au, T2u).copy()
    ur = ur.reshape(Au.shape[4] ** 2 * T2u.shape[1], Au.shape[5] ** 2 * T1.shape[3])
    # dr.T is needed in matrix product. Transpose *before* reshape to optimize copy
    dr = contract_dr_corner(Ad, T2d, T3, C3).transpose(3, 4, 5, 0, 1, 2).copy()
    dr = dr.reshape(Ad.shape[5] ** 2 * T3.shape[3], Ad.shape[2] ** 2 * T2d.shape[0])
    #      1-UR
    #         |
    #         0
    #         1
    #         |
    #      0-DR
    return dr @ ur


########################################################################################
#  construct 2x2 corners using U(1) symmetry
#  memory: peak at 2*chi**2*D**4
#
#  Nearly all the contraction has been put inside add_a_blockU1, only the corner C is
#  contracted outside since it does not exist in renormalize_T. To be able to use the
#  same add_a_blockU1, some leg swapping is necessary.
#
########################################################################################


# Function add_a_conj takes double layer tensor a = A-A* as input in the form of a
# SymmetricTensor, with merged bra and ket legs *and* legs merged in two directions as
# rows and as columns. To save memory, only 2 versions of a exsit, a_ul and a_ur. To
# contract dr and dl corenrs, the transpose of a_ul and a_ur are used (same storage,
# see ctm_environment).
def contract_ul_corner_U1(C1, T1, T4, a_ul, col_T1_r, col_T4_d):
    """
    Contract upper left corner using U(1) symmetry.
    """
    ul = C1 @ T4.reshape(T4.shape[0], T4.shape[1] ** 2 * T4.shape[3])
    ul = add_a_blockU1(
        T1.transpose(1, 2, 0, 3).reshape(T1.shape[1] ** 2, T1.shape[0], T1.shape[3]),
        ul.reshape(C1.shape[0], T4.shape[1] ** 2, T4.shape[3]),
        a_ul,
        col_T1_r,
        col_T4_d,
        return_blockwise=True,
    )
    return ul


def contract_ur_corner_U1(T2, C2, a_ur, T1, col_T2_d, col_T1_l):
    """
    Contract upper right corner using U(1) symmetry.
    """
    ur = C2 @ T1.reshape(T1.shape[0], T1.shape[1] ** 2 * T1.shape[3])
    # a_ur has swapped up and right legs:
    #  3
    # 1 2
    #  0
    # + need to swap T2 legs according to add_a_blockU1 conventions
    ur = add_a_blockU1(
        T2.transpose(2, 3, 1, 0).reshape(T2.shape[2] ** 2, T2.shape[1], T2.shape[0]),
        ur.reshape(C2.shape[0], T1.shape[1] ** 2, T1.shape[3]),
        a_ur,
        col_T2_d,
        col_T1_l,
        return_blockwise=True,
    )
    return ur


def contract_dr_corner_U1(a_dr, T2, T3, C3, col_T2_u, col_T3_l):
    """
    Contract down right corner using U(1) symmetry.
    """
    # a_dr is actually a_ul.T
    # to get a corner with convient leg ordering, a swap is made between T2 and T3, ie
    # add_a_blockU1 is used from the other side of the mirror (instead of a simple
    # rotation from dr to ul). T2 becomes up and T3 becomes left.
    dr = C3 @ T3.transpose(2, 0, 1, 3).reshape(C3.shape[1], -1)
    dr = add_a_blockU1(
        T2.transpose(2, 3, 0, 1).reshape(T2.shape[2] ** 2, T2.shape[0], T2.shape[1]),
        dr.reshape(C3.shape[0], T3.shape[0] ** 2, T3.shape[3]),
        a_dr,
        col_T2_u,
        col_T3_l,
        return_blockwise=True,
    )
    return dr.T


def contract_dl_corner_U1(T4, a_dl, C4, T3, col_T4_u, col_T3_r):
    """
    Contract down left corner using U(1) symmetry.
    """
    dl = T3.reshape(-1, C4.shape[1]) @ C4.T
    # a_dl = a_ur.T has swapped up and right legs:
    #  1
    # 3 0
    #  2
    # to get a corner with convient leg ordering, a swap is made between T3 and T4, ie
    # add_a_blockU1 is used from the other side of the mirror (instead of a simple
    # rotation from dl to ul). T4 stays left and T3 becomes up.
    dl = add_a_blockU1(
        dl.reshape(T3.shape[0] ** 2, T3.shape[2], C4.shape[0]),
        T4.swapaxes(0, 3).reshape(T4.shape[3], T4.shape[1] ** 2, T4.shape[0]),
        a_dl,
        col_T3_r,
        col_T4_u,
        return_blockwise=True,
    )
    return dl.T


@numba.njit(parallel=True)
def fill_swapaxes(ul, row_indices, col_indices):
    dr = ul.shape[2]
    dc = ul.shape[3]
    m = np.empty((row_indices.size, col_indices.size))
    for i in numba.prange(row_indices.size):
        r0, r1 = divmod(row_indices[i], dr)
        for j in numba.prange(col_indices.size):
            c0, c1 = divmod(col_indices[j], dc)
            m[i, j] = ul[r0, c0, r1, c1]
    return m


@numba.njit
def swapaxes_reduce(ul, col_irreps, a_block_irreps, a_col_irreps):
    # combine ul.swapaxes(1,2) and U(1) block reduction
    # all the information on ul row_irreps is already in a_block col_irreps
    blocks = []
    block_irreps = []
    for irr in a_block_irreps:
        ci = (col_irreps == irr).nonzero()[0]
        if ci.size:
            ri = (a_col_irreps == irr).nonzero()[0]
            m = fill_swapaxes(ul, ri, ci)  # parallel
            blocks.append(m)
            block_irreps.append(irr)
    return blocks, block_irreps


def add_a_blockU1(up, left, a_block, col_up_r, col_left_d, return_blockwise=False):
    """
    Contract up and left then add blockwise a = AA* using U(1) symmetry.
    Use this function in both contract_corner_U1 and renormalize_T_U1.

    Parameters
    ----------
    up: (d0, d1, d2) ndarray
        Tensor on the upper side of AA*. Bra and ket legs are merged and leg conventions
        differ from standard clockwise order, see notes.
    left: (d2, d3, d4) ndarray
        Tensor on the right side of AA*. Common leg ordering, merged bra and ket legs.
    a_block: (d5 * d6, d0 * d3) U1_SymmetricTensor
        Contracted A-A* as a U1_SymmetricTensor, with right and down legs merged as rows
        and up and left merged as columns.
    col_up_r: (d1,) integer ndarray
        up tensor right irreps.
    col_left_d: (d4,) integer ndarray
        left tensor down irreps.
    return_blockwise: bool, optional
        Whether to cast the result to array.

    Returns
    -------
    ul: U1_SymmetricTensor / ndarray depending on return_blockwise, shape
        (d5 * d1, d6 * d4)
        Contracted tensor network.

    Notes
    -----
    Bra and ket legs are necesseraly merged in AA*, so for simplicity they must be
    merged in all other input tensors. Therefore a reshape has been called before this
    function, which may require a copy. To avoid an additional copy here, legs are
    assumed to be in convenient order for contraction. This makes no change for left
    tensor but requires a swap of 0 (right) and 1 (down) axes for up tensor.
     0          3-up-2              45
     |            ||                ||
     left=1,2     01             67=AA*=0,1
     |                              ||
     3                              23
    """
    #  --------up-2
    #  |       ||
    #  2       01
    #  0
    #  |
    #  left=1,2 -> 3,4
    #  |
    #  3 -> 5
    ul = (
        up.reshape(up.shape[0] * up.shape[1], up.shape[2])
        @ left.reshape(left.shape[0], left.shape[1] * left.shape[2])
    ).reshape(up.shape[0], up.shape[1], left.shape[1], left.shape[2])

    row_irreps = a_block.get_column_representation()
    col_irreps = a_block.combine_representations(col_up_r, col_left_d)
    blocks, block_irreps = swapaxes_reduce(
        ul, col_irreps, a_block.block_irreps, row_irreps
    )

    rep_ul = (
        a_block.axis_reps[4],
        a_block.axis_reps[5],
        -col_up_r,
        a_block.axis_reps[6],
        a_block.axis_reps[7],
        -col_left_d,
    )
    sh = tuple(a.size for a in rep_ul)
    ul = U1_SymmetricTensor.from_array(ul.reshape(sh), rep_ul, 2)

    #  --------up-4
    #  |       ||
    #  |       01
    #  left=2,3
    #  |
    #  5
    ul = ul.permutate((0, 1, 3, 4), (2, 5))
    ul = a_block @ ul
    #  -----up-4 -> 2
    #  |    ||
    #  left=AA*=0,1
    #  |    ||
    #  5    23 -> 3,4
    ul = ul.permutate((0, 1, 4), (2, 3, 5))

    if return_blockwise:
        return ul
    return ul.toarray().reshape(ul.matrix_shape)
