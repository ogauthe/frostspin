###############################################################################
#  construct 2x2 corners
#  memory: peak at 2*a*d*chi**2*D**4
###############################################################################


def contract_enlarged_corner(C1, T1, T4, A):
    r"""
    Contract a corner C-T//T-A. Take upper left corner as template.
    To avoid calling permute twice, assume convenient leg ordering

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
    ul = ul.permute((0, 3), (1, 4, 2, 5))
    ul = A @ ul

    #  C1----T1-6
    #  |     ||
    #  |     |2
    #  |   0 |        2 4
    #  |    \|         \|
    #  T4----A-4      5-A*-0
    #  | \3  |\         |\
    #  7     5 1        1 3
    ul = ul.permute((0, 1, 4, 5), (2, 3, 6, 7))  # memory peak 2*a*d*chi**2*D**4
    ul = A.permute((0, 1, 4, 5), (2, 3)).dagger() @ ul

    #  C1-T1-4 ---->2
    #  |  ||
    #  T4=AA*=2,0->0,1
    #  |  ||
    #  5  31
    #  5  34
    return ul.permute((2, 0, 4), (3, 1, 5))


def contract_ul(C1, T1, T4, A):
    return contract_enlarged_corner(
        C1,
        T1.permute((1, 2, 0), (3,)),
        T4.permute((0,), (1, 2, 3)),
        A.permute((0, 1, 3, 4), (2, 5)),
    )


def contract_ur(T1, C2, A, T2):
    return contract_enlarged_corner(
        C2,
        T2.permute((2, 3, 1), (0,)),
        T1.permute((0,), (1, 2, 3)),
        A.permute((0, 1, 4, 5), (3, 2)),
    )


def contract_dr(A, T2, T3, C3):
    """
    unusual leg convention: dr is transposed compared to clockwise order
    """
    return contract_enlarged_corner(
        C3.transpose(),
        T3.permute((0, 1, 3), (2,)),
        T2.permute((1,), (2, 3, 0)),
        A.permute((0, 1, 5, 2), (4, 3)),
    )


def contract_dl(T4, A, C4, T3):
    return contract_enlarged_corner(
        C4,
        T4.permute((1, 2, 0), (3,)),
        T3.permute((3,), (0, 1, 2)),
        A.permute((0, 1, 2, 3), (5, 4)),
    )


def contract_C1234(C1, C2, C4, C3):
    up = C2 @ C1
    down = C4 @ C3.transpose()
    out = up.full_contract(down)
    return out


def contract_T1T3(C1, T1, C2, C4, T3, C3):
    left = C1 @ C4
    T1p = T1.permute((0, 1, 2), (3,))
    left = T1p @ left
    #  C1--T1-0
    #  |   ||
    #  |   12
    #  C4-3

    right = C3.transpose() @ C2
    T3p = T3.permute((3, 0, 1), (2,))
    right = T3p @ right
    #       3-C2
    #     12   |
    #     ||   |
    #  0--T3--C3
    right = right.permute((0,), (3, 1, 2))

    out = left.full_contract(right)
    return out


def contract_T2T4(C1, C2, T4, T2, C4, C3):
    left = C2 @ C1
    T4 = T4.permute((0,), (1, 2, 3))
    left = left @ T4
    down = C4 @ C3.transpose()
    left = left.permute((0, 1, 2), (3,))
    left = left @ down
    T2 = T2.permute((1,), (0, 2, 3))
    out = left.full_contract(T2)
    return out


def contract_norm(C1, T1, C2, T4, A, T2, C4, T3, C3):
    ul = contract_ul(C1, T1, T4, A)
    #   C1--T1--2
    #   |   ||
    #   T4==AA==0,1
    #   |   ||
    #   5   34

    right = C2.transpose() @ T2.permute((0,), (2, 3, 1))
    right = right.permute((3,), (1, 2, 0))
    up = right @ ul
    #   C1--T1--C2
    #   |   ||   |
    #   T4==AA==T2
    #   |   ||   |
    #   3   12   0

    down = C3 @ T3.permute((2,), (0, 1, 3))
    down = down.permute((3,), (1, 2, 0))
    down = C4 @ down
    down = down.permute((1, 2, 0), (3,))
    out = up.full_contract(down)
    return out
