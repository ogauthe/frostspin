import numpy as np

from ctm_contract import (
    contract_ul_corner,
    contract_ur_corner,
    contract_dl_corner,
    contract_dr_corner,
)
from toolsU1 import BlockMatrixU1


def rdm_1x1(C1, T1, C2, T4, A, T2, C4, T3, C3):
    """
    Compute 1-site reduced density matrix from CTMRG environment tensors
    """
    #   C1-0       3-T1-0         1-C2
    #   |            ||              |
    #   1            12              0
    #         0        0
    #   0      \ 2      \ 2          0
    #   |       \|       \|          |
    #   T4-1   5-A--3   5-A*-1    2-T2
    #   | \2     |\       |\      3/ |
    #   3        4 1      4 1        1
    #
    #   0           01               0
    #   |           ||               |
    #   C4-1      3-T3-2          1-C3

    # bypassing tensordot makes code conceptually simpler and memory efficient but
    # unreadable
    rdm = np.tensordot(T4, C1, ((0,), (1,)))
    rdm = np.tensordot(rdm, C4, ((2,), (0,)))
    rdm = np.tensordot(rdm, T1, ((2,), (3,)))
    rdm = np.tensordot(rdm, A, ((4, 0), (2, 5)))
    dr = np.tensordot(T2, C2, ((0,), (0,)))
    dr = np.tensordot(dr, C3, ((0,), (0,)))
    dr = np.tensordot(dr, T3, ((3,), (2,)))
    rdm = np.tensordot(dr, rdm, ((2, 0, 3, 5), (2, 6, 7, 1)))
    rdm = np.tensordot(rdm, A.conj(), ((3, 2, 0, 1, 5), (2, 5, 3, 4, 1)))
    return rdm / np.trace(rdm)


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

    left = np.tensordot(C1, np.tensordot(T4, C4, ((3,), (0,))), ((1,), (0,)))
    # |----0   <= put chi in leg 0 and let it there
    # |  ||
    # |  12
    # |=3,4
    # |-5
    left = np.tensordot(T1l, left, ((3,), (0,)))
    # |----0
    # |  ||
    # |  24
    # |=3,5
    # |-1
    left = left.transpose(0, 5, 2, 4, 1, 3).copy()
    left = np.tensordot(left, Al, ((4, 5), (2, 5)))
    left = left.transpose(0, 1, 4, 6, 7, 2, 3, 5).copy()
    left = np.tensordot(left, Al.conj(), ((5, 6, 7), (2, 5, 1)))
    left = left.transpose(
        0, 2, 3, 5, 6, 7, 4, 1
    ).copy()  # exchange bra-ket to optimize copy
    left = np.tensordot(left, T3l, ((5, 6, 7), (1, 0, 3)))
    left = left.transpose(1, 3, 0, 2, 4, 5).copy()

    right = np.tensordot(C2, T1r, ((1,), (0,)))
    right = right.swapaxes(0, 3).copy()  # put chi as leg 0
    right = np.tensordot(right, np.tensordot(T2, C3, ((1,), (0,))), ((3,), (0,)))
    right = right.transpose(0, 5, 2, 4, 1, 3).copy()
    temp = Ar.transpose(2, 3, 1, 0, 5, 4).copy()
    right = np.tensordot(right, temp, ((4, 5), (0, 1)))
    right = right.transpose(0, 1, 5, 6, 7, 2, 3, 4).copy()  # 1st memory peak
    temp = Ar.transpose(2, 3, 1, 0, 5, 4).conj().copy()
    right = np.tensordot(right, temp, ((5, 6, 7), (0, 1, 2)))
    right = right.transpose(0, 2, 3, 5, 6, 7, 4, 1).copy()  # 2nd memory peak
    right = np.tensordot(right, T3r, ((5, 6, 7), (1, 0, 2)))
    right = right.transpose(0, 2, 4, 5, 1, 3).copy()

    rdm = np.tensordot(left, right, ((2, 3, 4, 5), (0, 1, 2, 3)))
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


def rdm_2x2(C1, T1l, T1r, C2, T4u, Aul, Aur, T2u, T4d, Adl, Adr, T2d, C4, T3l, T3r, C3):
    """
    Compute reduced density matrix on a 2x2 plaquette.
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

    # memory use: 3*chi**2*D**4*d**4

    ul = np.tensordot(T1l, C1, ((3,), (0,)))
    ul = np.tensordot(ul, T4u, ((3,), (0,)))
    ul = np.tensordot(ul, Aul, ((1, 3), (2, 5)))
    ul = np.tensordot(ul, Aul.conj(), ((1, 2, 5), (2, 5, 1)))
    dl = np.tensordot(T4d, C4, ((3,), (0,)))
    dl = np.tensordot(dl, T3l, ((3,), (3,)))
    dl = np.tensordot(dl, Adl, ((1, 3), (5, 4)))
    dl = np.tensordot(dl, Adl.conj(), ((1, 2, 5), (5, 4, 1)))
    left = np.tensordot(dl, ul, ((0, 3, 6), (1, 4, 7)))
    del ul, dl
    left = left.transpose(5, 7, 9, 2, 4, 0, 1, 3, 6, 8).copy()
    ur = np.tensordot(C2, T1r, ((1,), (0,)))
    ur = np.tensordot(ur, T2u, ((0,), (0,)))
    ur = np.tensordot(ur, Aur, ((0, 4), (2, 3)))
    ur = np.tensordot(ur, Aur.conj(), ((0, 3, 5), (2, 3, 1)))
    dr = np.tensordot(T2d, C3, ((1,), (0,)))
    dr = np.tensordot(dr, T3r, ((3,), (2,)))
    dr = np.tensordot(dr, Adr, ((1, 3), (3, 4)))
    dr = np.tensordot(dr, Adr.conj(), ((1, 2, 5), (3, 4, 1)))
    right = np.tensordot(dr, ur, ((3, 6, 0), (3, 6, 1)))
    del ur, dr
    right = right.transpose(1, 3, 6, 8, 5, 7, 9, 2, 4, 0).copy()  # reduce memory
    rdm = np.tensordot(right, left, ((4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5)))
    d4 = Aul.shape[0] * Aur.shape[0] * Adl.shape[0] * Adr.shape[0]
    rdm = rdm.transpose(6, 2, 4, 0, 7, 3, 5, 1).reshape(d4, d4)

    rdm /= np.trace(rdm)
    return rdm


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
    ul = np.tensordot(C1, T4u, ((1,), (0,)))
    # |----0   <= put chi in leg 0 and let it there
    # |  ||
    # |  12
    # |=3,4
    # |-5
    ul = np.tensordot(T1l, ul, ((3,), (0,)))
    # |----0
    # |  ||
    # |  24
    # |=3,5
    # |-1
    ul = ul.transpose(0, 5, 2, 4, 1, 3).copy()
    ul = np.tensordot(ul, Aul, ((4, 5), (2, 5)))
    ul = ul.transpose(0, 1, 4, 6, 7, 2, 3, 5).copy()  # mem 2*a*d*chi**2*D**4
    ul = np.tensordot(ul, Aul.conj(), ((5, 6, 7), (2, 5, 1)))
    #   ------0
    #   | 2 ||
    #   |  \||
    #   |======3,6
    #   |  /||
    #   | 5 ||
    #   1   47
    ul = ul.transpose(0, 3, 6, 2, 5, 1, 4, 7).reshape(
        T1l.shape[0] * Aul.shape[3] ** 2 * Aul.shape[0] ** 2,
        T4u.shape[3] * Aul.shape[4] ** 2,
    )  # mem 2*d**2*chi**2*D**4
    if dl is None:
        dl = contract_dl_corner(T4d, Adl, C4, T3l)
    elif isinstance(dl, BlockMatrixU1):
        dl = dl.toarray()
    rdm = ul @ dl  # mem (2*d**2+1)*chi**2*D**4
    rdm = rdm.reshape(T1l.shape[0] * Aul.shape[3] ** 2, Aul.shape[0] ** 2 * dl.shape[1])
    del ul, dl
    if ur is None:
        ur = contract_ur_corner(T1r, C2, Aur, T2u)
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

    #       0
    #  1,2 =|
    #       3
    dr = np.tensordot(T2d, C3, ((1,), (0,)))
    #       0
    #  1,2 =|
    #       |
    #   34  |
    # 5-||---
    dr = np.tensordot(dr, T3r, ((3,), (2,)))
    dr = dr.transpose(0, 5, 2, 4, 1, 3).copy()
    dr = np.tensordot(dr, Adr.swapaxes(0, 1), ((4, 5), (3, 4)))
    #     6    0            3    0
    #     | 5  |            | 2  |
    #     |/ 2-|            |/ 5-|
    #   7------|   --->   4------|
    #     |3 \ |            |6 \ |
    #     ||  4|            ||  7|
    #   1-------          1-------
    dr = dr.transpose(
        0, 1, 5, 6, 7, 2, 3, 4
    ).copy()  # memory peak: chi**2*D**4*(d**2+2*a*d)
    dr = np.tensordot(dr, Adr.conj(), ((5, 6, 7), (3, 4, 1)))
    #     36   0
    #     || 2 |
    #     ||/  |              0
    # 4,7======|   ---->      |
    #     ||\  |            22|
    #     || 5 |          1----
    #   1-------
    dr = dr.transpose(0, 3, 6, 1, 4, 7, 2, 5).reshape(
        rdm.shape[1], Adr.shape[0] ** 2
    )  # memory peak: 3*d**2*chi**2*D**4
    rdm = rdm @ dr
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
    dr=None,
):
    """
    -------
    |  |02│
    |-----|
    |13|  |
    -------
    memory: 3*d**2*chi**2*D**4
    """
    dl = np.tensordot(T4d, C4, ((3,), (0,)))
    dl = np.tensordot(dl, T3l, ((3,), (3,)))
    #  0             0
    #  |-1           |-5
    #  |-2      -->  |-3
    #  |  34         |  42
    #  |  ||         |  ||
    #  ------5       -------1
    dl = dl.transpose(0, 5, 4, 2, 3, 1).copy()
    dl = np.tensordot(dl, Adl.swapaxes(0, 1), ((4, 5), (4, 5)))
    #  0   6          0   3
    #  |   | 5        |   | 2
    #  |-3 |/         |-5 |/
    #  |---|--7  -->  |---|--4
    #  |  2|\         |  6|\
    #  |  || 4        |  || 7
    #  ------1        ------1
    dl = dl.transpose(0, 1, 5, 6, 7, 2, 3, 4).copy()  # 2*d*a*chi**2*D**4
    dl = np.tensordot(dl, Adl.conj(), ((5, 6, 7), (4, 5, 1)))
    #  0  36          0  12
    #  |2 ||          |3 ||
    #  | \||          | \||
    #  |====4,7  -->  |====6,7
    #  | /||          | /||
    #  |5 ||          |4 ||
    #  ------1        ------5
    dl = dl.transpose(0, 3, 6, 2, 5, 1, 4, 7).reshape(
        T4d.shape[0] * Adl.shape[2] ** 2,
        Adl.shape[0] ** 2 * T3l.shape[2] * Adl.shape[3] ** 2,
    )  # 2*d**2*chi**2*D**4
    if ul is None:
        ul = contract_ul_corner(C1, T1l, T4u, Aul)
    elif isinstance(ul, BlockMatrixU1):
        ul = ul.toarray()
    rdm = ul @ dl  # (2*d**2+1)*chi**2*D**4
    rdm = rdm.reshape(ul.shape[0] * Adl.shape[0] ** 2, T3l.shape[2] * Adl.shape[3] ** 2)
    del ul, dl
    if dr is None:
        dr = contract_dr_corner(Adr, T2d, T3r, C3)
    elif isinstance(dr, BlockMatrixU1):
        dr = dr.toarray()
    rdm = rdm @ dr.T  # (2*d**2+1)*chi**2*D**4
    rdm = rdm.reshape(T1l.shape[0] * Aul.shape[3] ** 2, Adl.shape[0] ** 2, dr.shape[0])
    del dr
    # ---0          ---1
    # |   2   -->   |   2
    # |11 |         |00 |        mem 2*d**2*chi**2*D**4
    # -----         -----
    rdm = rdm.swapaxes(0, 1).reshape(
        Adl.shape[0] ** 2,
        T1l.shape[0] * Aul.shape[3] ** 2 * T2d.shape[0] * Adr.shape[2] ** 2,
    )

    ur = np.tensordot(C2, T1r, ((1,), (0,)))
    ur = ur.swapaxes(0, 3).copy()  # put chi as leg 0
    ur = np.tensordot(ur, T2u, ((3,), (0,)))
    ur = ur.transpose(0, 3, 2, 5, 1, 4).copy()
    temp = Aur.transpose(2, 3, 1, 0, 4, 5).copy()
    ur = np.tensordot(ur, temp, ((4, 5), (0, 1)))
    ur = ur.transpose(0, 1, 5, 6, 7, 2, 3, 4).copy()  # (d**2+2*a*d)*chi**2*D**4
    temp = Aur.transpose(2, 3, 1, 0, 4, 5).conj().copy()
    ur = np.tensordot(ur, temp, ((5, 6, 7), (0, 1, 2)))
    #   0--------
    #       || 2|
    #       ||/ |
    #  4,7======|   -->   0----   memory peak: 2*ur + rdm
    #       ||\ |           22|               = 3*d**2*chi**2*D**4
    #       || 5|             |
    #       36  1             1
    ur = ur.transpose(0, 4, 7, 1, 3, 6, 2, 5).reshape(rdm.shape[1], Aur.shape[0] ** 2)
    rdm = rdm @ ur
    rdm = rdm.reshape(Adl.shape[0], Adl.shape[0], Aur.shape[0], Aur.shape[0])
    rdm = rdm.transpose(2, 0, 3, 1).reshape(
        Aur.shape[0] * Adl.shape[0], Aur.shape[0] * Adl.shape[0]
    )
    rdm /= rdm.trace()
    return rdm
