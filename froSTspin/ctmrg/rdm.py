def rdm_1x1(C1, T1, C2, T4, A, T2, C4, T3, C3):
    C3p = C3.transpose()
    T2p = T2.permute((1,), (0, 2, 3))
    right = C3p @ T2p
    right = right.permute((0, 2, 3), (1,))
    right = right @ C2
    T3p = T3.permute((3,), (0, 1, 2))
    down = C4 @ T3p
    T1p = T1.permute((0, 1, 2), (3,))
    rdm = T1p @ C1
    T4p = T4.permute((0,), (1, 2, 3))
    rdm = rdm @ T4p
    rdm = rdm.permute((0, 2, 4, 5), (1, 3))
    Ap = A.permute((2, 5), (0, 1, 3, 4))
    rdm = rdm @ Ap
    down = down.permute((0, 1, 2), (3,))
    right = right.permute((0,), (1, 2, 3))
    down = down @ right
    down = down.permute((2, 4), (5, 3, 0, 1))
    rdm = rdm.permute((0, 6, 3, 7), (1, 2, 4, 5))
    down = down @ rdm
    down = down.permute((4,), (2, 3, 1, 0, 5))
    Aconj = A.permute((0,), (2, 5, 3, 4, 1)).dagger()
    rdm = down @ Aconj

    rdm = rdm.toarray()
    rdm /= rdm.trace()
    return rdm


def contract_open_corner(C1, T1, T4, A):
    ul = T1.permute((1, 2, 0), (3,))
    ul = ul @ C1
    ul = ul @ T4.permute((0,), (1, 2, 3))

    # |----2        |-----4
    # |  ||         |   ||
    # |  01    -->  |   02
    # |=3,4         |=1,3
    # 5             5
    ul = ul.permute((0, 3), (1, 4, 2, 5))

    #  C1----T1-4       C1----T1-6
    #  |     ||         |     ||
    #  |     02         |     |4
    #  |   1 4          |   1 |
    #  |    \|    -->   |    \|
    #  T4-15-A-2        T4----A-2
    #  | \3  |\         | \5  |\
    #  5     3 0        7     3 0
    ul = A.permute((1, 0, 3, 4), (2, 5)) @ ul
    ul = ul.permute((0, 4, 5), (1, 2, 3, 6, 7))  # memory a*d*chi**2*D**4

    #  C1----T1-6
    #  |     ||
    #  |     |1
    #  |   3 |        0 4
    #  |    \|         \|
    #  T4----A-4      5-A*-1
    #  | \2  |\         |\
    #  7     5 0        2 3
    ul = A.permute((1, 2, 5), (0, 3, 4)).dagger() @ ul
    ul = ul.permute((3, 0, 4, 1, 6), (5, 2, 7))
    # -----6          -----4
    # |  || 3         |  || 0
    # |  ||/          |  ||/
    # |=====4,1  -->  |=====2,3
    # |  ||\          |  ||\
    # 7  52 0         7  56 1
    return ul


def contract_open_corner_mirror(T1, C2, A, T2):
    # rdm_1x2 and rdm_diag_dr require different output convention here
    # cannot provide clean leg ordering output
    ur = C2 @ T1.permute((0,), (1, 2, 3))
    ur = ur.transpose() @ T2.permute((0,), (2, 3, 1))
    ur = ur.permute((0, 3), (1, 4, 2, 5))
    ur = A.permute((1, 0, 4, 5), (2, 3)) @ ur
    ur = ur.permute((0, 4, 5), (1, 2, 3, 6, 7))
    ur = A.permute((1, 2, 3), (0, 4, 5)).dagger() @ ur
    #   6------
    #    2 || |
    #     \|| |
    #  5,3====|
    #     /|| |
    #    0 41 7
    return ur


def rdm_1x2(C1, T1l, T1r, C2, T4, Al, Ar, T2, C4, T3l, T3r, C3):
    """
    Compute reduced density matrix for 2 sites in a row
    asymmetric CPU: chi**2*D**6*(a*d + a*d**2) + d**2*chi**3*D**4 = O(D**10)
    asymmetric memory: 2*d**2*chi**2*D**4
    """
    #
    #   C1-0     3-T1-0           3-T1-0       1-C2
    #   |          ||               ||            |
    #   1          12               12            0
    #        0       0        0       0
    #   0     \ 2     \ 2      \ 2     \ 2        0
    #   |      \|      \|       \|      \|        |
    #   T4-1  5-A--3  5-A*-3   5-A--3  5-A*-3  2-T2
    #   | \2    |\      |\       |\      |\    3/ |
    #   3       4 1     4 1      4 1     4 1      1
    #
    #   0          01               01            0
    #   |          ||               ||            |
    #   C4-1     3-T3-2           3-T3-2       1-C3

    # no need to specify SymmetricTensor n_leg_rows, just call permute which may
    # have no effect.
    left = T4.permute((0, 1, 2), (3,))
    left = left @ C4
    left = contract_open_corner(C1, T1l, left, Al)
    left = left @ T3l.permute((0, 1, 3), (2,))
    left = left.permute((0, 1), (2, 3, 4, 5))
    # -----4
    # | 0
    # |  \
    # |===2,3
    # |  /
    # | 1
    # -----5

    right = T2.permute((0, 2, 3), (1,))
    right = right @ C3
    right = right.permute((0, 3), (1, 2))
    right = contract_open_corner_mirror(T1r, C2, Ar, right)
    right = right.permute((3, 0, 5, 2, 6), (4, 1, 7))
    right = right @ T3r.permute((0, 1, 2), (3,))
    right = right.permute((2, 3, 4, 5), (0, 1))
    rdm = left @ right
    rdm = rdm.permute((0, 2), (1, 3)).toarray(as_matrix=True)
    rdm /= rdm.trace()
    return rdm


def rdm_2x1(C1, T1, C2, T4u, Au, T2u, T4d, Ad, T2d, C4, T3, C3):
    """
    Compute reduced density matrix for 2 sites in a column
    """
    # contract using 1x2 with swapped tensors and legs
    return rdm_1x2(
        C2,
        T2u.permute((1, 2, 3), (0,)),
        T2d.permute((1,), (2, 3, 0)),
        C3.transpose(),
        T1,
        Au.permute((0, 1), (3, 4, 5, 2)),
        Ad.permute((0, 1), (3, 4, 5, 2)),
        T3.permute((2, 3, 0), (1,)),
        C1,
        T4u.permute((1, 2, 3), (0,)),
        T4d.permute((1, 2, 3), (0,)),
        C4.transpose(),
    )


def rdm_diag_dr(C1, T1l, ur, T4u, Aul, dl, Adr, T2d, T3r, C3):
    """
    -------
    |02|  |
    |-----|
    |  |13|
    -------
    memory: 3*d**2*chi**2*D**4
    """
    rdm = contract_open_corner(C1, T1l, T4u, Aul)
    #  ------4
    #  |  || 0
    #  |  ||/
    #  |==||=2,3
    #  |  ||\
    #  7  56 1
    rdm = rdm @ dl  # mem (2*d**2+1)*chi**2*D**4

    #  ------4
    #  |  || 0
    #  |  ||/
    #  |==||=2,3
    #  |  ||\
    #  |  || 1
    #  |  ||=5,6
    #  ------7
    rdm = rdm.permute((2, 3, 4), (0, 1, 5, 6, 7))
    rdm = ur @ rdm  # mem (2*d**2+1)*chi**2*D**4
    rdm = rdm.permute((3, 4), (0, 1, 2, 5, 6, 7))
    #   -----           -----
    #   |11 |   --->    |00 |
    #   |   0           |   1
    #   |--2            |--2

    dr = contract_open_corner_mirror(
        T2d.permute((1, 2, 3), (0,)),
        C3.transpose(),
        Adr.permute((0, 1, 3, 4), (5, 2)),
        T3r.permute((2, 3), (0, 1)),
    )
    #     23   4
    #     || 0 |
    #     ||/  |               1
    # 5,6======|   ---->       |
    #     ||\  |             00|
    #     || 1 |          1'----
    #   7-------
    dr = dr.permute((5, 2, 6, 4, 1, 7), (3, 0))  # memory peak: 3*d**2*chi**2*D**4
    rdm = rdm @ dr
    rdm = rdm.permute((0, 2), (1, 3)).toarray(as_matrix=True)
    rdm /= rdm.trace()
    return rdm


def rdm_diag_ur(ul, T1r, C2, Aur, T2u, T4d, Adl, dr_T, C4, T3l):
    """
    -------
    |  |02│
    |-----|
    |13|  |
    -------
    memory: 3*d**2*chi**2*D**4

    Note that dr corner is transposed compared to standard clockwise order.
    """
    rdm = rdm_diag_dr(
        C4,
        T4d,
        ul,
        T3l.permute((3,), (0, 1, 2)),
        Adl.permute((0, 1), (5, 2, 3, 4)),
        dr_T,  # transposed dr in input avoids to transpose it here
        Aur.permute((0, 1), (5, 2, 3, 4)),
        T1r.permute((3,), (0, 1, 2)),
        T2u.permute((2, 3, 0), (1,)),
        C2.transpose(),
    )
    rdm = (
        rdm.reshape(Adl.shape[0], Aur.shape[0], Adl.shape[0], Aur.shape[0])
        .transpose(1, 0, 3, 2)
        .reshape(Aur.shape[0] * Adl.shape[0], Aur.shape[0] * Adl.shape[0])
    )
    return rdm


# fmt: off
def rdm_2x2(C1, T1l, T1r, C2, T4u, Aul, Aur, T2u, T4d, Adl, Ard, T2d, C4, T4l, T4r, C3):  # noqa: ARG001, E501
    raise NotImplementedError("To do!")
