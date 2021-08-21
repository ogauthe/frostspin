from ctmrg.ctm_contract import contract_ur_corner, contract_dl_corner
from groups.block_matrix_U1 import BlockMatrixU1


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


def rdm_1x2(C1, T1l, T1r, C2, T4, Al, Ar, T2, C4, T3l, T3r, C3):
    """
    Compute reduced density matrix for 2 sites in a row
    CPU: chi**2*D**6*(a*d + a*d**2) + d**2*chi**3*D**4 = O(D**10)
    Memory: 2*d**2*chi**2*D**4
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

    left = T4.reshape(T4.shape[0] * T4.shape[1] ** 2, T4.shape[3]) @ C4
    left = left.reshape(T4.shape[0], T4.shape[1], T4.shape[1], C4.shape[1])
    left = contract_open_corner(C1, T1l, left, Al)
    left = left.copy().reshape(
        Al.shape[0] ** 2 * Al.shape[3] ** 2 * T1l.shape[0],
        Al.shape[4] ** 2 * C4.shape[1],
    )
    left = left @ T3l.swapaxes(2, 3).reshape(left.shape[1], T3l.shape[2])
    left = left.reshape(
        Al.shape[0] ** 2, Al.shape[3] ** 2 * T1l.shape[0] * T3l.shape[2]
    )
    # -----4
    # | 0
    # |  \
    # |===2,3
    # |  /
    # | 1
    # -----5

    right = T2.transpose(0, 2, 3, 1).reshape(
        T2.shape[0] * T2.shape[2] ** 2, T2.shape[1]
    )
    right = (
        (right @ C3)
        .reshape(T2.shape[0], T2.shape[2], T2.shape[2], C3.shape[1])
        .transpose(0, 3, 1, 2)
    )
    right = contract_open_corner_mirror(T1r, C2, Ar, right)
    right = right.copy().reshape(
        Ar.shape[0] ** 2 * Ar.shape[5] ** 2 * T1r.shape[3],
        Ar.shape[4] ** 2 * C3.shape[1],
    )
    right = right @ T3r.reshape(right.shape[1], T3r.shape[3])
    right = right.reshape(Ar.shape[0] ** 2, left.shape[1])

    rdm = (left @ right.T).reshape(Al.shape[0], Al.shape[0], Ar.shape[0], Ar.shape[0])
    rdm = rdm.swapaxes(1, 2).reshape(
        Al.shape[0] * Ar.shape[0], Al.shape[0] * Ar.shape[0]
    )
    rdm /= rdm.trace()
    return rdm


def rdm_2x1(C1, T1, C2, T4u, Au, T2u, T4d, Ad, T2d, C4, T3, C3):
    """
    Compute reduced density matrix for 2 sites in a column
    """
    # contract using 1x2 with swapped tensors and legs
    return rdm_1x2(
        C2,
        T2u.transpose(1, 2, 3, 0),
        T2d.transpose(1, 2, 3, 0),
        C3.T,
        T1,
        Au.transpose(0, 1, 3, 4, 5, 2),
        Ad.transpose(0, 1, 3, 4, 5, 2),
        T3.transpose(2, 3, 0, 1),
        C1,
        T4u.transpose(1, 2, 3, 0),
        T4d.transpose(1, 2, 3, 0),
        C4.T,
    )


def rdm_diag_dr(
    C1,
    T1l,
    T1r,
    C2,
    T4u,
    Aul,
    Aur,
    T2u,
    T4d,
    Adl,
    Adr,
    T2d,
    C4,
    T3l,
    T3r,
    C3,
    ur=None,
    dl=None,
):
    """
    -------
    |02|  |
    |-----|
    |  |13|
    -------
    memory: 3*d**2*chi**2*D**4
    """
    ul = contract_open_corner(C1, T1l, T4u, Aul).transpose(2, 3, 4, 0, 1, 5, 6, 7)
    #  ------4          ------2
    #  |  || 0          |  || 3
    #  |  ||/           |  ||/
    #  |==||=2,3    --> |==||=0,1
    #  |  ||\           |  ||\
    #  7  56 1          7  56 4
    ul = ul.copy().reshape(
        Aul.shape[3] ** 2 * T1l.shape[0] * Aul.shape[0] ** 2,
        Aul.shape[4] ** 2 * T4u.shape[3],
    )  # mem 2*d**2*chi**2*D**4

    if dl is None:
        dl = contract_dl_corner(T4d, Adl, C4, T3l).copy()
        dl = dl.reshape(
            Adl.shape[2] ** 2 * T4d.shape[0], Adl.shape[3] ** 2 * T3l.shape[2]
        )
    elif isinstance(dl, BlockMatrixU1):
        dl = dl.toarray()
    #     --0   1-
    #   1\|      |
    #  1'/|      0
    #     2
    #     0
    #     |
    #     -1
    rdm = ul @ dl  # mem (2*d**2+1)*chi**2*D**4
    rdm = rdm.reshape(Aul.shape[3] ** 2 * T1l.shape[0], Aul.shape[0] ** 2 * dl.shape[1])
    del ul, dl
    if ur is None:
        ur = contract_ur_corner(T1r, C2, Aur, T2u).copy()
        ur = ur.reshape(
            Aur.shape[4] ** 2 * T2u.shape[1], Aur.shape[5] ** 2 * T1r.shape[3]
        )
    elif isinstance(ur, BlockMatrixU1):
        ur = ur.toarray()
    rdm = ur @ rdm  # mem (2*d**2+1)*chi**2*D**4
    rdm = rdm.reshape(ur.shape[0], Aul.shape[0] ** 2, T3l.shape[2] * Adl.shape[3] ** 2)
    del ur
    #   -----           -----
    #   |11 |   --->    |00 |
    #   |   0           |   1
    #   |--2            |--2
    rdm = rdm.swapaxes(0, 1).reshape(
        Aul.shape[0] ** 2,
        T2u.shape[1] * Aur.shape[4] ** 2 * T3l.shape[2] * Adl.shape[3] ** 2,
    )  # mem 2*d**2*chi**2*D**4

    dr = contract_open_corner_mirror(
        T2d.transpose(1, 2, 3, 0),
        C3.T,
        Adr.transpose(0, 1, 3, 4, 5, 2),
        T3r.transpose(2, 3, 0, 1),
    )
    #     23   4
    #     || 0 |
    #     ||/  |               1
    # 5,6======|   ---->       |
    #     ||\  |             00|
    #     || 1 |          1'----
    #   7-------
    dr = dr.reshape(Adr.shape[0] ** 2, rdm.shape[1])  # memory peak: 3*d**2*chi**2*D**4
    rdm = rdm @ dr.T
    rdm = rdm.reshape(Aul.shape[0], Aul.shape[0], Adr.shape[0], Adr.shape[0])
    rdm = rdm.swapaxes(1, 2).reshape(
        Aul.shape[0] * Adr.shape[0], Aul.shape[0] * Adr.shape[0]
    )
    rdm /= rdm.trace()
    return rdm


def rdm_diag_ur(
    C1,
    T1l,
    T1r,
    C2,
    T4u,
    Aul,
    Aur,
    T2u,
    T4d,
    Adl,
    Adr,
    T2d,
    C4,
    T3l,
    T3r,
    C3,
    ul=None,
    dr_T=None,
):
    """
    -------
    |  |02│
    |-----|
    |13|  |
    -------
    memory: 3*d**2*chi**2*D**4

    Note that optional argument dr is transposed compared to standard clockwise order.
    """
    rdm = rdm_diag_dr(
        C4,
        T4d,
        T4u,
        C1,
        T3l.transpose(3, 0, 1, 2),
        Adl.transpose(0, 1, 5, 2, 3, 4),
        Aul.transpose(0, 1, 5, 2, 3, 4),
        T1l.transpose(3, 0, 1, 2),
        T3r.transpose(3, 0, 1, 2),
        Adr.transpose(0, 1, 5, 2, 3, 4),
        Aur.transpose(0, 1, 5, 2, 3, 4),
        T1r.transpose(3, 0, 1, 2),
        C3.T,
        T2d.transpose(2, 3, 0, 1),
        T2u.transpose(2, 3, 0, 1),
        C2.T,
        ur=ul,
        dl=dr_T,  # transposed dr in input avoids to transpose it here
    )
    rdm = (
        rdm.reshape(Adl.shape[0], Aur.shape[0], Adl.shape[0], Aur.shape[0])
        .transpose(1, 0, 3, 2)
        .copy()
        .reshape(Aur.shape[0] * Adl.shape[0], Aur.shape[0] * Adl.shape[0])
    )
    return rdm
