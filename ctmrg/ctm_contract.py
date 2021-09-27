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
#  Nearly all the contraction has been put inside add_a_bilayer, only the corner C is
#  contracted outside since it does not exist in renormalize_T. To be able to use the
#  same add_a_bilayer, some leg swapping is necessary.
#
########################################################################################


# Function add_a_conj takes double layer tensor a = A-A* as input in the form of a
# SymmetricTensor, with merged bra and ket legs *and* legs merged in two directions as
# rows and as columns. To save memory, only 2 versions of a exsit, a_ul and a_ur. To
# contract dr and dl corenrs, the transpose of a_ul and a_ur are used (same storage,
# see ctm_environment).
def contract_ul_corner_U1(C1, T1, T4, a_ul):
    """
    Contract upper left corner using U(1) symmetry.
    """
    ul = C1 @ T4.permutate((0,), (1, 2, 3))
    ul = add_a_bilayer(T1.permutate((1, 2, 0), (3,)), ul, a_ul)
    return ul


def contract_ur_corner_U1(T2, C2, a_ur, T1):
    """
    Contract upper right corner using U(1) symmetry.
    """
    ur = C2 @ T1.permutate((0,), (1, 2, 3))
    # a_ur has swapped up and right legs:
    #  3
    # 1 2
    #  0
    # + need to swap T2 legs according to add_a_bilayer conventions
    ur = add_a_bilayer(T2.permutate((2, 3, 1), (0,)), ur, a_ur)
    return ur


def contract_dr_corner_U1(a_dr, T2, T3, C3):
    """
    Contract down right corner using U(1) symmetry.
    """
    # a_dr is actually a_ul.T
    # to get a corner with convient leg ordering, a swap is made between T2 and T3, ie
    # add_a_bilayer is used from the other side of the mirror (instead of a simple
    # rotation from dr to ul). T2 becomes up and T3 becomes left.
    up = T2.permutate((2, 3, 0), (1,))
    left = C3 @ T3.permutate((2,), (0, 1, 3))
    dr = add_a_bilayer(up, left, a_dr)
    return dr.T


def contract_dl_corner_U1(T4, a_dl, C4, T3):
    """
    Contract down left corner using U(1) symmetry.
    """
    dl = T3.permutate((0, 1, 2), (3,)) @ C4.T
    # a_dl = a_ur.T has swapped up and right legs:
    #  1
    # 3 0
    #  2
    # to get a corner with convient leg ordering, a swap is made between T3 and T4, ie
    # add_a_bilayer is used from the other side of the mirror (instead of a simple
    # rotation from dl to ul). T4 stays left and T3 becomes up.

    left = T4.permutate((3,), (1, 2, 0))
    dl = add_a_bilayer(dl, left, a_dl)
    return dl.T


def add_a_bilayer(up, left, a_ul):
    """
    Contract up and left then add blockwise a = AA* using U(1) symmetry.
    Use this function in both corner contraction and T renormalization.

    For simplicity, consider corner ul in notations and diagrams. Can be used with
    other corners as long as leg ordering fits.

    Parameters
    ----------
    up: SymmetricTensor
        Tensor on the upper side of AA*. Non clockwise leg ordering, see notes for
        conventions.
    left: SymmetricTensor
        Tensor on the right side of AA*. Clockwise leg ordering.
    a_ul: SymmetricTensor
        Contracted A-A* as a SymmetricTensor.

    Returns
    -------
    ul: SymmetricTensor
        Contracted tensor network.

    Notes
    -----
    To avoid needless copy here, legs are assumed to be in convenient order for
    contraction. This makes no change for left tensor but requires a swap of 0 (right)
    and 1 (down) axes for up tensor. Up and left must be contracted, ie the last leg of
    up corresponds to columns and the first leg of left corresponds to rows.

     0          3-up-2               45
     |            ||                 ||
     left=1,2     01            6,7=a_ul=0,1
     |                               ||
     3                               23
    """
    #  --------up-2
    #  |       ||
    #  2       01
    #  0
    #  |
    #  left=1,2 -> 3,4
    #  |
    #  3 -> 5
    ul = up @ left
    #  --------up-4
    #  |       ||
    #  |       01
    #  left=2,3
    #  |
    #  5
    ul = ul.permutate((0, 1, 3, 4), (2, 5))
    ul = a_ul @ ul
    #  ------up--4 -> 2
    #  |     ||
    #  left=a_ul=0,1
    #  |     ||
    #  5     23 -> 3,4
    ul = ul.permutate((0, 1, 4), (2, 3, 5))
    return ul
