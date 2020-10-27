"""
Corner and halves contraction for CTMRG algorithm
Library agnostic module, only calls __matmul__, reshape and transpose methods.
"""

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

    #  C1-T1-4 ---->0
    #  |  ||
    #  T4=AA=2,0->1,2
    #  |  ||
    #  5  31
    #  3  45
    return ul  # do not reshape to avoid copy here


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
    ).transpose(3, 0, 6, 4, 1, 5, 2, 7)
    # -----2
    # | 0
    # |  \
    # |=====3,4
    # |  /||
    # 7 1 56
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
    ur = ur.transpose(3, 0, 6, 5, 2, 4, 1, 7)
    #   2------
    #    0    |
    #     \   |
    #  3,4====|
    #     /|| |
    #    1 56 7
    return ur


def contract_ul_corner(C1, T1, T4, A):
    return contract_corner(C1, T1, T4, A).transpose(4, 2, 0, 5, 3, 1)


def contract_ur_corner(T1, C2, A, T2):
    return contract_corner(
        C2, T2.transpose(1, 2, 3, 0), T1, A.transpose(0, 1, 3, 4, 5, 2)
    ).transpose(4, 2, 0, 5, 3, 1)


def contract_dr_corner(A, T2, T3, C3):
    return contract_corner(
        C3.T,
        T3.transpose(3, 0, 1, 2),
        T2.transpose(1, 2, 3, 0),
        A.transpose(0, 1, 4, 5, 2, 3),
    ).transpose(5, 3, 1, 4, 2, 0)


def contract_dl_corner(T4, A, C4, T3):
    return contract_corner(
        C4, T4, T3.transpose(3, 0, 1, 2), A.transpose(0, 1, 5, 2, 3, 4)
    ).transpose(4, 2, 0, 5, 3, 1)


###############################################################################
# construct halves from corners
# memory: max during corner construction
###############################################################################


def contract_u_half(C1, T1l, T1r, C2, T4, Al, Ar, T2):
    ul = contract_ul_corner(C1, T1l, T4, Al).copy()
    ul = ul.reshape(T1l.shape[0] * Al.shape[3] ** 2, T4.shape[3] * Al.shape[4] ** 2)
    ur = contract_ur_corner(T1r, C2, Ar, T2).copy()
    ur = ur.reshape(T2.shape[1] * Ar.shape[4] ** 2, T1r.shape[3] * Ar.shape[5] ** 2)
    #  UL-01-UR
    #  |      |
    #  1      0
    return ur @ ul


def contract_l_half(C1, T1, T4u, Au, T4d, Ad, C4, T3):
    ul = contract_ul_corner(C1, T1, T4u, Au).copy()
    ul = ul.reshape(T1.shape[0] * Au.shape[3] ** 2, T4u.shape[3] * Au.shape[4] ** 2)
    dl = contract_dl_corner(T4d, Ad, C4, T3).copy()
    dl = dl.reshape(T4d.shape[0] * Ad.shape[2] ** 2, T3.shape[2] * Ad.shape[3] ** 2)
    #  UL-0
    #  |
    #  1
    #  0
    #  |
    #  DL-1
    return ul @ dl


def contract_d_half(T4, Al, Ar, T2, C4, T3l, T3r, C3):
    dl = contract_dl_corner(T4, Al, C4, T3l).copy()
    dl = dl.reshape(T4.shape[0] * Al.shape[2] ** 2, T3l.shape[2] * Al.shape[3] ** 2)
    dr = contract_dr_corner(Ar, T2, T3r, C3).transpose(3, 4, 5, 0, 1, 2).copy()
    dr = dr.reshape(T3r.shape[3] * Ar.shape[5] ** 2, T2.shape[0] * Ar.shape[2] ** 2)
    #  0      1
    #  0      0
    #  |      |
    #  DL-11-DR
    return dl @ dr


def contract_r_half(T1, C2, Au, T2u, Ad, T2d, T3, C3):
    ur = contract_ur_corner(T1, C2, Au, T2u).copy()
    ur = ur.reshape(T2u.shape[1] * Au.shape[4] ** 2, T1.shape[3] * Au.shape[5] ** 2)
    dr = contract_dr_corner(Ad, T2d, T3, C3).transpose(3, 4, 5, 0, 1, 2).copy()
    dr = dr.reshape(T3.shape[3] * Ad.shape[5] ** 2, T2d.shape[0] * Ad.shape[2] ** 2)
    #      1-UR
    #         |
    #         0
    #         1
    #         |
    #      0-DR
    return dr @ ur
