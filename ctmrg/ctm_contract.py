"""
Corner and halves contraction for CTMRG algorithm
Library agnostic module, only calls __matmul__, reshape and transpose methods.
"""
import numpy as np

from groups.toolsU1 import combine_colors
from groups.block_matrix_U1 import BlockMatrixU1

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


def contract_open_corner(C1, T1, T4, A):
    ul = T1.transpose(1, 2, 0, 3).reshape(T1.shape[1] ** 2 * T1.shape[0], T1.shape[3])
    ul = ul @ C1
    ul = ul @ T4.reshape(T4.shape[0], T4.shape[1] ** 2 * T4.shape[3])
    ul = ul.reshape(
        T1.shape[1], T1.shape[2], T1.shape[0], T4.shape[1], T4.shape[2], T4.shape[3]
    )

    # |----2        |-----4
    # |  ||         |   ||
    # |  01    -->  |   02
    # |=3,4         |=1,3
    # 5             5
    ul = (
        ul.transpose(0, 3, 1, 4, 2, 5)
        .copy()
        .reshape(
            T1.shape[1] * T4.shape[1],
            T1.shape[2] * T4.shape[2] * T1.shape[0] * T4.shape[3],
        )
    )

    temp = A.transpose(1, 0, 3, 4, 2, 5).reshape(
        A.shape[1] * A.shape[0] * A.shape[3] * A.shape[4], A.shape[2] * A.shape[5]
    )

    #  C1----T1-4
    #  |     ||
    #  |     02
    #  |   1 4
    #  |    \|
    #  T4-15-A-2
    #  | \3  |\
    #  5     3 0
    ul = (temp @ ul).reshape(
        A.shape[1],
        A.shape[0] * A.shape[3] * A.shape[4],
        T1.shape[2] * T4.shape[2],
        T1.shape[0] * T4.shape[3],
    )
    ul = (
        ul.swapaxes(1, 2)
        .copy()
        .reshape(
            A.shape[1] * T1.shape[2] * T4.shape[2],
            A.shape[0] * A.shape[3] * A.shape[4] * T1.shape[0] * T4.shape[3],
        )
    )  # memory 2*a*d*chi**2*D**4

    #  C1----T1-6
    #  |     ||
    #  |     |1
    #  |   3 |        0 4
    #  |    \|         \|
    #  T4----A-4      5-A*-1
    #  | \2  |\         |\
    #  7     5 0        2 3
    temp = (
        temp.reshape(
            A.shape[1], A.shape[0] * A.shape[3] * A.shape[4], A.shape[2] * A.shape[5]
        )
        .swapaxes(0, 1)
        .conj()
        .copy()
        .reshape(
            A.shape[0] * A.shape[3] * A.shape[4], A.shape[1] * A.shape[2] * A.shape[5]
        )
    )
    ul = temp @ ul
    ul = ul.reshape(
        A.shape[0],
        A.shape[3],
        A.shape[4],
        A.shape[0],
        A.shape[3],
        A.shape[4],
        T1.shape[0],
        T4.shape[3],
    ).transpose(3, 0, 4, 1, 6, 5, 2, 7)
    # -----6          -----4
    # |  || 3         |  || 0
    # |  ||/          |  ||/
    # |=====4,1  -->  |=====2,3  (no copy, non-contiguous array)
    # |  ||\          |  ||\
    # 7  52 0         7  56 1
    return ul


def contract_open_corner_mirror(T1, C2, A, T2):
    ur = T2.transpose(0, 2, 3, 1).reshape(T2.shape[0], T2.shape[2] ** 2 * T2.shape[1])
    ur = (C2 @ T1.reshape(T1.shape[0], T1.shape[1] ** 2 * T1.shape[3])).T @ ur
    temp = (
        A.transpose(1, 0, 4, 5, 2, 3)
        .copy()
        .reshape(
            A.shape[1] * A.shape[0] * A.shape[4] * A.shape[5], A.shape[2] * A.shape[3]
        )
    )
    ur = (
        ur.reshape(
            T1.shape[1], T1.shape[2], T1.shape[3], T2.shape[2], T2.shape[3], T2.shape[1]
        )
        .transpose(0, 3, 1, 4, 2, 5)
        .copy()
        .reshape(temp.shape[1], temp.shape[1] * T1.shape[3] * T2.shape[1])
    )
    ur = (temp @ ur).reshape(
        A.shape[1],
        A.shape[0] * A.shape[4] * A.shape[5],
        temp.shape[1],
        T1.shape[3] * T2.shape[1],
    )
    ur = (
        ur.swapaxes(1, 2)
        .copy()
        .reshape(
            A.shape[1] * temp.shape[1],
            A.shape[0] * A.shape[4] * A.shape[5] * T1.shape[3] * T2.shape[1],
        )
    )
    temp = (
        temp.reshape(
            A.shape[1], A.shape[0] * A.shape[4] * A.shape[5], A.shape[2] * A.shape[3]
        )
        .swapaxes(0, 1)
        .conj()
        .copy()
        .reshape(A.shape[0] * A.shape[4] * A.shape[5], ur.shape[0])
    )
    ur = (temp @ ur).reshape(
        A.shape[0],
        A.shape[4],
        A.shape[5],
        A.shape[0],
        A.shape[4],
        A.shape[5],
        T1.shape[3],
        T2.shape[1],
    )
    ur = ur.transpose(3, 0, 5, 2, 6, 4, 1, 7)
    #   4------
    #    0 || |
    #     \|| |
    #  2,3====|
    #     /|| |
    #    1 56 7
    return ur


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
# BlockMatrixU1, with merged bra and ket legs *and* legs merged in two directions as
# rows and as columns. To save memory, only 2 versions of a exsit, a_ul and a_ur. To
# contract dr and dl corenrs, the transpose of a_ul and a_ur are used (same storage,
# see ctm_environment).
def contract_ul_corner_U1(
    C1, T1, T4, a_ul, col_T1_r, col_T4_d, col_a_ul, col_a_r, col_a_d
):
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
        col_a_ul,
        col_a_r,
        col_a_d,
        return_blockwise=True,
    )
    return ul


def contract_ur_corner_U1(
    T2, C2, a_ur, T1, col_T2_d, col_a_ur, col_a_d, col_a_l, col_T1_l
):
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
        col_a_ur,
        col_a_d,
        col_a_l,
        return_blockwise=True,
    )
    return ur


def contract_dr_corner_U1(
    a_dr, T2, T3, C3, col_a_rd, col_a_u, col_a_l, col_T2_u, col_T3_l
):
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
        col_a_rd,
        col_a_u,
        col_a_l,
        return_blockwise=True,
    )
    return dr.T


def contract_dl_corner_U1(
    T4, a_dl, C4, T3, col_T4_u, col_a_dl, col_a_u, col_a_r, col_T3_r
):
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
        col_a_dl,
        col_a_r,
        col_a_u,
        return_blockwise=True,
    )
    return dl.T


def swapaxes12_toblock(M, row_colors, col_colors, d1, d2):
    row_sort = row_colors.argsort(kind="mergesort")
    sorted_row_colors = row_colors[row_sort]
    col_sort = col_colors.argsort(kind="mergesort")
    sorted_col_colors = col_colors[col_sort]
    row_blocks = (
        [0]
        + list((sorted_row_colors[:-1] != sorted_row_colors[1:]).nonzero()[0] + 1)
        + [row_colors.size]
    )
    col_blocks = (
        [0]
        + list((sorted_col_colors[:-1] != sorted_col_colors[1:]).nonzero()[0] + 1)
        + [col_colors.size]
    )

    blocks = []
    block_colors = []
    row_indices = []
    col_indices = []
    rbi, cbi, rbimax, cbimax = 0, 0, len(row_blocks) - 1, len(col_blocks) - 1
    d3 = col_colors.size // d2

    while rbi < rbimax and cbi < cbimax:
        if sorted_row_colors[row_blocks[rbi]] == sorted_col_colors[col_blocks[cbi]]:
            ri = row_sort[row_blocks[rbi] : row_blocks[rbi + 1]].copy()
            ci = col_sort[col_blocks[cbi] : col_blocks[cbi + 1]].copy()
            row_indices.append(ri)  # copy ri to own data and delete row_sort at exit
            col_indices.append(ci)  # same for ci
            ri0, ri1 = np.divmod(ri, d1)
            ci0, ci1 = np.divmod(ci, d2)
            ri_swap = (ri0[:, None] * d3 + ci0).ravel()
            ci_swap = (ri1[:, None] * d2 + ci1).ravel()
            m = M[ri_swap, ci_swap].reshape(ri.size, ci.size)
            # m = np.empty((ri.size, ci.size))
            # for i, r in enumerate(ri):
            #    r0, r1 = divmod(r, d1)
            #    for j, c in enumerate(ci):
            #        c0, c1 = divmod(c, d2)
            #        m[i,j] = t[r0, c0, r1, c1]

            blocks.append(m)
            block_colors.append(sorted_row_colors[row_blocks[rbi]])
            rbi += 1
            cbi += 1
        elif sorted_row_colors[row_blocks[rbi]] < sorted_col_colors[col_blocks[cbi]]:
            rbi += 1
        else:
            cbi += 1
    return block_colors, blocks, row_indices, col_indices


def add_a_blockU1(
    up,
    left,
    a_block,
    col_up_r,
    col_left_d,
    col_a_ul,
    col_a_r,
    col_a_d,
    return_blockwise=False,
):
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
    a_block: (d5 * d6, d0 * d3) BlockMatrixU1
      Contracted A-A* as a BlockMatrixU1, with right and down legs merged as rows and up
      and left merged as columns.
    col_up_r: (d1,) integer ndarray
      up tensor right colors.
    col_left_d: (d4,) integer ndarray
      left tensor down colors.
    col_a_ul: (d0 * d3,) integer ndarray
      a_block column colors, corresponding to merged up and left legs.
    col_a_r: (d5,) integer ndarray
      a_block right colors
    col_a_d: (d6,) integer ndarray
      a_block down colors
    return_blockwise: bool, optional
      Whether to cast the result into BlockMatrixU1.

    Returns
    -------
    ul: BlockMatrixU1 / ndarray depending on return_blockwise, shape (d5 * d1, d6 * d4)
      Contracted tensor network.

    Notes
    -----
    Bra and ket legs are necesseraly merged in AA*, so for simplicity they must be
    merged in all other input tensors. Therefore a reshape has been called before this
    function, which may require a copy. To avoid an additional copy here, legs are
    assumed to be in convenient order for contraction. This makes no change for left
    tensor but requires a swap of 0 (right) and 1 (down) axes for up tensor.
     0        2-up-1              2
     |          ||                ||
     left=1     0               3=AA*=0
     |                            ||
     2                             1
    """
    #  --------up-1
    #  |       ||
    #  2       0
    #  0
    #  |
    #  left=1 -> 2
    #  |
    #  2 -> 3
    ul = up.reshape(up.shape[0] * up.shape[1], up.shape[2]) @ left.reshape(
        left.shape[0], left.shape[1] * left.shape[2]
    )

    ########################################################################
    # => need a dedidacted U(1) function here
    # send ul to BlockMatrixU1 *while* swapping axes
    # contract with a_block
    # send back to dense while swapping again

    #  --------up-2
    #  |       ||
    #  |       0
    #  left=1
    #  |
    #  3
    cc = combine_colors(col_up_r, col_left_d)
    ul = BlockMatrixU1(
        (col_a_ul.size, cc.size),
        *swapaxes12_toblock(ul, col_a_ul, cc, left.shape[1], left.shape[2])
    )
    ul = a_block @ ul
    # reshape through dense casting. This is inefficient.
    #  -----up-2 -> 1
    #  |    ||
    #  left=AA*=0
    #  |    ||
    #  3    1 -> 2
    ul = ul.toarray().reshape(col_a_r.size, col_a_d.size, up.shape[1], left.shape[2])
    ul = ul.swapaxes(1, 2).reshape(
        col_a_r.size * up.shape[1], col_a_d.size * left.shape[2]
    )
    ###########################################################################
    if return_blockwise:
        rc = combine_colors(col_a_r, col_up_r)
        cc = -combine_colors(col_a_d, col_left_d)
        ul = BlockMatrixU1.from_dense(ul, rc, cc)
    return ul
