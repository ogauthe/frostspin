###############################################################################
#  construct 2x2 corners
#  memory: peak at 2*a*d*chi**2*D**4
###############################################################################


def contract_enlarged_corner(C1, T1, T4, A):
    r"""
    Contract a corner C-T//T-A. Take upper left corner as template.
    To avoid calling permutate twice, assume convenient leg ordering

     0          3-T1-2      0 4
     |            ||         \|
     T4=1,2       01        5-A-2
     |                        |\
     3                        3 1
    """
    #  C1-03-T1-2
    #  |     ||
    #  1->3  01
    ul = T1 @ C1

    #  C1---T1-2
    #  |    ||
    #  3    01
    #  0
    #  |
    #  T4=1,2 -> 3,4
    #  |
    #  3 -> 5
    ul = ul @ T4

    #  C1----T1-4
    #  |     ||
    #  |     02
    #  |   0 4
    #  |    \|
    #  T4-15-A-2
    #  | \3  |\
    #  5     3 1
    ul = ul.permutate((0, 3), (1, 4, 2, 5))
    ul = A @ ul

    #  C1----T1-6
    #  |     ||
    #  |     |2
    #  |   0 |        2 4
    #  |    \|         \|
    #  T4----A-4      5-A*-0
    #  | \3  |\         |\
    #  7     5 1        1 3
    ul = ul.permutate((0, 1, 4, 5), (2, 3, 6, 7))  # memory peak 2*a*d*chi**2*D**4
    ul = A.permutate((0, 1, 4, 5), (2, 3)).dagger() @ ul

    #  C1-T1-4 ---->2
    #  |  ||
    #  T4=AA*=2,0->0,1
    #  |  ||
    #  5  31
    #  5  34
    return ul.permutate((2, 0, 4), (3, 1, 5))


def contract_ul(C1, T1, T4, A):
    return contract_enlarged_corner(
        C1,
        T1.permutate((1, 2, 0), (3,)),
        T4.permutate((0,), (1, 2, 3)),
        A.permutate((0, 1, 3, 4), (2, 5)),
    )


def contract_ur(T1, C2, A, T2):
    return contract_enlarged_corner(
        C2,
        T2.permutate((2, 3, 1), (0,)),
        T1.permutate((0,), (1, 2, 3)),
        A.permutate((0, 1, 4, 5), (3, 2)),
    )


def contract_dr(A, T2, T3, C3):
    """
    unusual leg convention: dr is transposed compared to clockwise order
    """
    return contract_enlarged_corner(
        C3.transpose(),
        T3.permutate((0, 1, 3), (2,)),
        T2.permutate((1,), (2, 3, 0)),
        A.permutate((0, 1, 5, 2), (4, 3)),
    )


def contract_dl(T4, A, C4, T3):
    return contract_enlarged_corner(
        C4,
        T4.permutate((1, 2, 0), (3,)),
        T3.permutate((3,), (0, 1, 2)),
        A.permutate((0, 1, 2, 3), (5, 4)),
    )
