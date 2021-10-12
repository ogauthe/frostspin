###############################################################################
#  construct 2x2 corners
#  memory: peak at 2*a*d*chi**2*D**4
###############################################################################


def contract_corner_monolayer(C1, T1, T4, A):
    """
    Contract a corner C-T//T-A. Take upper left corner as template.
    """
    #  C1-03-T1-2
    #  |     ||
    #  1->3  01
    ul = T1.permutate((1, 2, 0), (3,))
    ul = ul @ C1

    #  C1---T1-2
    #  |    ||
    #  3    01
    #  0
    #  |
    #  T4=1,2 -> 3,4
    #  |
    #  3 -> 5
    ul = ul @ T4.permutate((0,), (1, 2, 3))

    #  C1----T1-4
    #  |     ||
    #  |     02
    #  |   0 4
    #  |    \|
    #  T4-15-A-2
    #  | \3  |\
    #  5     3 1
    ul = ul.permutate((0, 3), (1, 4, 2, 5))
    ul = A.permutate((0, 1, 3, 4), (2, 5)) @ ul

    #  C1----T1-6
    #  |     ||
    #  |     |2
    #  |   0 |        2 4
    #  |    \|         \|
    #  T4----A-4      5-A*-0
    #  | \3  |\         |\
    #  7     5 1        1 3
    ul = ul.permutate((0, 1, 4, 5), (2, 3, 6, 7))  # memory peak 2*a*d*chi**2*D**4
    ul = A.permutate((3, 4), (0, 1, 2, 5)).conjugate() @ ul

    #  C1-T1-4 ---->2
    #  |  ||
    #  T4=AA*=2,0->0,1
    #  |  ||
    #  5  31
    #  5  34
    return ul.permutate((2, 0, 4), (3, 1, 5))


def contract_ul_corner_monolayer(C1, T1, T4, A):
    return contract_corner_monolayer(C1, T1, T4, A)


def contract_ur_corner_monolayer(T1, C2, A, T2):
    return contract_corner_monolayer(
        C2, T2.permutate((1, 2, 3), (0,)), T1, A.permutate((0, 1), (3, 4, 5, 2))
    )


def contract_dr_corner_monolayer(A, T2, T3, C3):
    """
    unusual leg convention: dr is transposed compared to clockwise order
    """
    return contract_corner_monolayer(
        C3.T,
        T3.permutate((3,), (0, 1, 2)),
        T2.permutate((1,), (2, 3, 0)),
        A.permutate((0, 1), (4, 5, 2, 3)),
    )


def contract_dl_corner_monolayer(T4, A, C4, T3):
    return contract_corner_monolayer(
        C4, T4, T3.permutate((3,), (0, 1, 2)), A.permutate((0, 1), (5, 2, 3, 4))
    )


########################################################################################
#  construct 2x2 corners bilayer
#  memory: peak at 2*chi**2*D**4
#
#  Nearly all the contraction has been put inside add_a_bilayer, only the corner C is
#  contracted outside since it does not exist in renormalize_T. To be able to use the
#  same add_a_bilayer, some leg swapping is necessary.
#
########################################################################################


# Function add_a_bilayer takes double layer tensor a = A-A* as input in the form of a
# SymmetricTensor, with merged bra and ket legs *and* legs merged in two directions as
# rows and as columns. To save memory, only 2 versions of a exsit, a_ul and a_ur. To
# contract dr and dl corenrs, the transpose of a_ul and a_ur are used (same storage,
# see ctm_environment).
def contract_ul_corner_bilayer(C1, T1, T4, a_ul):
    """
    Contract upper left corner using contracted A-A*
    """
    ul = C1 @ T4.permutate((0,), (1, 2, 3))
    ul = add_a_bilayer(T1.permutate((1, 2, 0), (3,)), ul, a_ul)
    return ul


def contract_ur_corner_bilayer(T1, C2, a_ur, T2):
    """
    Contract upper right corner using contracted A-A*
    """
    ur = C2 @ T1.permutate((0,), (1, 2, 3))
    # a_ur has swapped up and right legs:
    #  3
    # 1 2
    #  0
    # + need to swap T2 legs according to add_a_bilayer conventions
    ur = add_a_bilayer(T2.permutate((2, 3, 1), (0,)), ur, a_ur)
    return ur


def contract_dr_corner_bilayer(a_dr, T2, T3, C3):
    """
    Contract down right corner using contracted A-A*
    """
    # a_dr is actually a_ul.T
    # to get a corner with convient leg ordering, a swap is made between T2 and T3, ie
    # add_a_bilayer is used from the other side of the mirror (instead of a simple
    # rotation from dr to ul). T2 becomes up and T3 becomes left.
    up = T2.permutate((2, 3, 0), (1,))
    left = C3 @ T3.permutate((2,), (0, 1, 3))
    dr = add_a_bilayer(up, left, a_dr)
    return dr.T


def contract_dl_corner_bilayer(T4, a_dl, C4, T3):
    """
    Contract down left corner using contracted A-A*
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
