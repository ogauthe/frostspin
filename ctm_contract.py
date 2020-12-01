"""
Corner and halves contraction for CTMRG algorithm
Library agnostic module, only calls __matmul__, reshape and transpose methods.
"""

from toolsU1 import combine_colors, BlockMatrixU1
from toolsU1 import checkU1

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
    # -----4
    # |  || 0
    # |  ||/
    # |=====2,3
    # |  ||\
    # 7  56 1
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


###############################################################################
#  construct 2x2 corners using U(1) symmetry
#  memory: peak at 2*chi**2*D**4
###############################################################################


def contract_ul_corner_U1(
    C1, T1, T4, a_ul, colors_T1_r, colors_T4_d, colors_a_ul, col_a_r, col_a_d
):
    """
    Contract upper left corner using U(1) symmetry.
    """
    ul = C1 @ T4.reshape(T4.shape[0], T4.shape[1] ** 2 * T4.shape[3])
    ul = ul.reshape(C1.shape[0], T4.shape[1], T4.shape[2], T4.shape[3])
    ul = add_a_blockU1(T1, ul, a_ul, colors_a_ul, colors_T1_r, colors_T4_d)

    # reshape through dense casting. This is inefficient.
    ul = ul.toarray().reshape(col_a_r.size, col_a_d.size, T1.shape[0], T4.shape[3])
    #  C1-T1-2 -> 1
    #  |  ||
    #  T4=AA*=0
    #  |  ||
    #  3  1 -> 2
    ul = ul.swapaxes(1, 2).reshape(
        col_a_r.size * T1.shape[0], col_a_d.size * T4.shape[3]
    )
    rc = combine_colors(col_a_r, colors_T1_r)
    cc = -combine_colors(col_a_d, colors_T4_d)
    print("contract_ul_corner_U1", checkU1(ul, (rc, -cc)))
    ul = BlockMatrixU1.from_dense(ul, rc, cc)
    return ul


def contract_ur_corner_U1(
    T2, C2, a_ur, T1, colors_T2_d, colors_a_ur, col_a_d, col_a_l, colors_T1_l
):
    """
    Contract upper right corner using U(1) symmetry.
    """
    ur = C2 @ T1.reshape(T1.shape[0], T1.shape[1] ** 2 * T1.shape[3])
    ur = ur.reshape(C2.shape[0], T1.shape[1], T1.shape[2], T1.shape[3])
    # a_ur has swapped up and right legs:
    #  3
    # 1 2
    #  0
    ur = add_a_blockU1(
        T2.transpose(1, 2, 3, 0), ur, a_ur, colors_a_ur, colors_T2_d, colors_T1_l
    )

    # reshape through dense casting. This is inefficient.
    ur = ur.toarray().reshape(col_a_d.size, col_a_l.size, T2.shape[1], T1.shape[3])
    #  3-T1---
    #    ||  |
    #  1=AA*=|
    #    ||  |
    #     0  2
    ur = ur.swapaxes(1, 2).reshape(
        col_a_d.size * T2.shape[1], col_a_l.size * T1.shape[3]
    )
    rc = combine_colors(col_a_d, colors_T2_d)
    cc = -combine_colors(col_a_l, colors_T1_l)
    print("contract_ur_corner_U1", checkU1(ur, (rc, -cc)))
    ur = BlockMatrixU1.from_dense(ur, rc, cc)
    return ur


def contract_dr_corner_U1(
    a_dr, T2, T3, C3, colors_a_rd, col_a_u, col_a_l, colors_T2_u, colors_T3_l
):
    """
    Contract down right corner using U(1) symmetry.
    """
    dr = C3.T @ T2.swapaxes(0, 1).reshape(T2.shape[1], -1)
    dr = dr.reshape(C3.shape[1], T2.shape[0], T2.shape[2], T2.shape[3])
    dr = add_a_blockU1(
        dr.transpose(1, 2, 3, 0),  # MIRROR
        T3.transpose(2, 0, 1, 3),
        a_dr,
        -colors_a_rd,
        -colors_T2_u,
        -colors_T3_l,
    )

    # reshape through dense casting. This is inefficient.
    dr = dr.toarray().reshape(col_a_u.size, col_a_l.size, T2.shape[0], T3.shape[3])
    #     0  2
    #    ||  |
    #  1=AA*=|
    #    ||  |
    #  3------
    dr = dr.swapaxes(1, 2).reshape(
        col_a_u.size * T2.shape[0], col_a_l.size * T3.shape[3]
    )
    rc = combine_colors(col_a_u, colors_T2_u)
    cc = -combine_colors(col_a_l, colors_T3_l)
    print("contract_dr_corner_U1", checkU1(dr, (rc, -cc)))
    dr = BlockMatrixU1.from_dense(dr, rc, cc)
    return dr.T


def contract_dl_corner_U1(
    T4, a_dl, C4, T3, colors_T4_u, colors_a_dl, col_a_u, col_a_r, colors_T3_r
):
    """
    Contract down left corner using U(1) symmetry.
    """
    dl = T4.reshape(-1, T4.shape[3]) @ C4
    dl = dl.reshape(T4.shape[0], T4.shape[1], T4.shape[2], C4.shape[1])
    # a_dl has swapped up and right legs:
    #  1
    # 3 0
    #  2
    # MIRROR in add_a_block
    dl = add_a_blockU1(
        T3.transpose(2, 0, 1, 3),
        dl.transpose(3, 1, 2, 0),
        a_dl,
        -colors_a_dl,
        -colors_T3_r,
        -colors_T4_u,
    )

    # reshape through dense casting. This is inefficient.
    dl = dl.toarray().reshape(col_a_r.size, col_a_u.size, T3.shape[2], T4.shape[0])
    #  3  1
    #  | ||
    #  |=AA*=0
    #  | ||
    #  ------2
    dl = dl.swapaxes(1, 2).reshape(
        col_a_r.size * T3.shape[2], col_a_u.size * T4.shape[0]
    )
    rc = combine_colors(col_a_r, colors_T3_r)
    cc = -combine_colors(col_a_u, colors_T4_u)
    print("contract_dl_corner_U1", checkU1(dl, (rc, -cc)))
    dl = BlockMatrixU1.from_dense(dl, rc, cc)
    return dl.T


def add_a_blockU1(T1, C1T4, a_block, colors_a_ul, colors_T1_r, colors_T4_d):
    """
    Use this function in both contract_corner_U1 and renormalize_T_U1.
    """
    ul = T1.transpose(1, 2, 0, 3).reshape(T1.shape[1] ** 2 * T1.shape[0], T1.shape[3])

    #  -----T1-2
    #  |    ||
    #  3    01
    #  0
    #  |
    #  T4=1,2 -> 3,4
    #  |
    #  3 -> 5
    ul = (ul @ C1T4.reshape(C1T4.shape[0], C1T4.shape[1] ** 2 * C1T4.shape[3])).reshape(
        T1.shape[1],
        T1.shape[2],
        T1.shape[0],
        C1T4.shape[1],
        C1T4.shape[2],
        C1T4.shape[3],
    )
    ul = (
        ul.transpose(0, 1, 3, 4, 2, 5)
        .copy()
        .reshape(T1.shape[1] ** 2 * C1T4.shape[1] ** 2, T1.shape[0] * C1T4.shape[3])
    )
    #  -----T1-4
    #  |    01
    #  |
    #  T4=2,3
    #  |
    #  5
    col_col = combine_colors(colors_T1_r, colors_T4_d)
    print("add_a_block", checkU1(ul, (colors_a_ul, -col_col)))
    ul1 = BlockMatrixU1.from_dense(ul, colors_a_ul, col_col)
    ul2 = a_block @ ul1
    print(
        "add_a_block",
        ((ul2.toarray() - a_block.toarray() @ ul) ** 2).sum() ** 0.5 / ul2.norm(),
    )
    #  C1-T1-4
    #  |  ||
    #  T4=AA*=0,1
    #  |  ||
    #  5  23
    # cannot reshape (neither transpose) here since left and down dimensions are unknown
    # let contract_corner_U1 and renormalize_T_U1 decide what to do next
    return ul2
