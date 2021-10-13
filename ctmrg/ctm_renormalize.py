import numpy as np

from ctmrg.ctm_contract import add_a_bilayer


def construct_projectors(
    corner1, corner2, corner3, corner4, chi, rcutoff, degen_ratio, window
):

    R = corner1 @ corner2
    Rt = corner3 @ corner4
    M = R @ Rt
    U, s, V = M.svd(cut=chi, window=window, rcutoff=rcutoff, degen_ratio=degen_ratio)
    del M

    for bi, sbi in enumerate(s):
        s12 = 1.0 / np.sqrt(sbi)
        U.blocks[bi][:] *= s12
        V.blocks[bi][:] *= s12[:, None]

    P = R.T @ U.conjugate()
    Pt = Rt @ V.H
    return P, Pt


###############################################################################
# conventions: renormalize_C(C,T,P)
#              renormalize_T(Pt,T,A,P)
###############################################################################


def renormalize_T(Pt, T, A, P):
    """
    Renormalize edge T using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    #           0 2
    #            \|
    #         0 5-A-3 0
    #        /    |\   \
    #     1-P     4 1   Pt-1
    #       \     01    /
    #        \    ||   /
    #         0'3-T3-20'
    nT = T.permutate((0, 1, 2), (3,))
    nT = nT @ P.permutate((2,), (0, 1, 3))
    nT = nT.permutate((0, 3), (1, 4, 2, 5))
    nT = A.permutate((0, 1, 2, 3), (4, 5)) @ nT
    nT = nT.permutate((0, 1, 4, 5), (2, 3, 6, 7))
    nT = A.permutate((2, 3), (0, 1, 4, 5)).conjugate() @ nT
    nT = nT.permutate((3, 1, 4), (2, 0, 5))
    nT = Pt.T @ nT
    nT /= nT.norm()
    return nT


def renormalize_corner_P(C, T, P):
    """
    Renormalize corner C using projector P
    CPU: 2*chi**3*D**2
    """
    # assume axes are already adjusted.

    #           0
    #         1-|
    #         2-|
    #           3
    #           0
    #         1-|
    nC = T @ C
    #           0
    #       /-1-|
    # 1-P=01--2-|
    #       \   |
    #        \3-|
    nC = nC.permutate((0,), (1, 2, 3))
    nC = nC @ P
    nC /= nC.norm()
    return nC


def renormalize_corner_Pt(C, T, Pt):
    """
    Renormalize corner C using projector Pt
    CPU: 2*chi**3*D**2
    """
    # assume axes are already adjusted.

    #  0
    #  |-1
    #  |-2
    #  3
    #  0
    #  |-1
    nC = T @ C
    #  0
    #  |-1-\
    #  |-2--10=P-1
    #  |   /
    #  |-3/
    nC = nC.permutate((0,), (1, 2, 3))
    nC = nC @ Pt
    nC /= nC.norm()
    return nC


def renormalize_C1_up(C1, T4, P):
    """
    Renormalize corner C1 from an up move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C1.T, T4.permutate((3, 1, 2), (0,)), P).T


def renormalize_T1_monolayer(Pt, T1, A, P):
    """
    Renormalize edge T1 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    nT1 = renormalize_T(
        Pt, T1.permutate((1, 2, 3), (0,)), A.permutate((0, 1), (4, 5, 2, 3)), P
    )
    return nT1.permutate((3,), (1, 2, 0))


def renormalize_C2_up(C2, T2, Pt):
    """
    Renormalize corner C2 from an up move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C2, T2.permutate((1, 2, 3), (0,)), Pt)


def renormalize_C2_right(C2, T1, P):
    """
    Renormalize corner C2 from right move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C2.T, T1.permutate((3, 1, 2), (0,)), P).T


def renormalize_T2_monolayer(Pt, T2, A, P):
    """
    Renormalize edge T2 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    nT2 = renormalize_T(
        Pt, T2.permutate((2, 3, 0), (1,)), A.permutate((0, 1), (5, 2, 3, 4)), P
    )
    return nT2.permutate((0,), (3, 1, 2))


def renormalize_C3_right(C3, T3, Pt):
    """
    Renormalize corner C3 from right move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C3.T, T3.permutate((3, 0, 1), (2,)), Pt).T


def renormalize_C3_down(C3, T2, P):
    """
    Renormalize corner C3 from down move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C3, T2.permutate((0, 2, 3), (1,)), P)


def renormalize_T3_monolayer(Pt, T3, A, P):
    """
    Renormalize edge T3 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    return renormalize_T(Pt, T3, A, P).permutate((1, 2, 0), (3,))


def renormalize_C4_down(C4, T4, Pt):
    """
    Renormalize corner C4 from a down move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C4, T4.permutate((0, 1, 2), (3,)), Pt)


def renormalize_C4_left(C4, T3, P):
    """
    Renormalize corner C4 from a left move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C4.T, T3.permutate((2, 0, 1), (3,)), P).T


def renormalize_T4_monolayer(Pt, T4, A, P):
    """
    Renormalize edge T4 using projectors P and Pt
    CPU: 2*chi**2*D**4*(a*d*D**2 + chi)
    """
    nT4 = renormalize_T(
        Pt, T4.permutate((1, 2, 3), (0,)), A.permutate((0, 1), (3, 4, 5, 2)), P
    )
    return nT4.permutate((3,), (1, 2, 0))


def renormalize_C1_left(C1, T1, Pt):
    """
    Renormalize corner C1 from a left move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C1, T1.permutate((0, 1, 2), (3,)), Pt)


###############################################################################
# U(1) symmetric renormalize_T
###############################################################################


def renormalize_T1_bilayer(Pt, T1, a_ul, P):
    """
    Renormalize edge T1 using projectors P and Pt and bilayer A-A* tensor
    CPU: highly depends on symmetry, worst case chi**2*D**8
    """
    # Pt -> left, need swapaxes
    # T1 -> up, transpose due to add_a_bilayer conventions
    nT1 = T1.permutate((1, 2, 0), (3,))
    left = Pt.permutate((2,), (0, 1, 3))
    nT1 = add_a_bilayer(nT1, left, a_ul)
    #             -T1-0'
    #            / ||
    #       1'-Pt==AA=0
    #            \ ||
    #               1'
    nT1 = P.T @ nT1
    nT1 /= nT1.norm()
    return nT1


def renormalize_T2_bilayer(Pt, T2, a_ur, P):
    """
    Renormalize edge T2 using projectors P and Pt and bilayer A-A* tensor
    CPU: highly depends on symmetry, worst case chi**2*D**8
    """
    # Pt -> left, need swapaxes
    # T2 -> up
    nT2 = T2.permutate((2, 3, 1), (0,))
    left = Pt.permutate((2,), (0, 1, 3))
    nT2 = add_a_bilayer(nT2, left, a_ur)
    #                 3
    #                 |
    #                Pt
    #              //  |
    #           2==AA=T2
    #              \\  |
    #               0  1
    nT2 = P.T @ nT2
    nT2 /= nT2.norm()
    nT2 = nT2.permutate((3,), (0, 1, 2))
    return nT2


def renormalize_T3_bilayer(Pt, T3, a_dl, P):
    """
    Renormalize edge T3 using projectors P and Pt and bilayer A-A* tensor
    CPU: highly depends on symmetry, worst case chi**2*D**8
    """
    #             1
    #             ||
    #         0 3-AA-0 0
    #        /    ||   \
    #     1-P      2   Pt-1
    #       \     01    /
    #        \    ||   /
    #         0'3-T3-20'
    # A mirror is needed to use a_dl, swap Pt and P
    # P -> left (contract with leg 3 of a_dl)
    # T3 -> up
    nT3 = T3.permutate((0, 1, 2), (3,))
    left = P.permutate((2,), (0, 1, 3))
    nT3 = add_a_bilayer(nT3, left, a_dl)
    #               2
    #              ||
    #            //AA=0
    #         3-P  ||
    #            \-T3-1
    nT3 = Pt.T @ nT3
    nT3 /= nT3.norm()
    nT3 = nT3.permutate((1, 2, 0), (3,))
    return nT3


def renormalize_T4_bilayer(Pt, T4, a_ul, P):
    """
    Renormalize edge T4 using projectors P and Pt and bilayer A-A* tensor
    CPU: highly depends on symmetry, worst case chi**2*D**8
    """
    # we can use either a_ul or a_dl. In both cases, the second projector must be added
    # with nT4 = nT4 @ P / Pt due to a leg ordering. Use a_ul to have down leg in 3.
    # P -> up
    # T4 -> left
    nT4 = P.permutate((0, 1, 3), (2,))
    left = T4.permutate((0,), (1, 2, 3))
    nT4 = add_a_bilayer(nT4, left, a_ul)
    #              1
    #              |
    #              P
    #            / ||
    #          T4==AA=0
    #            \ ||
    #            3  2
    nT4 = nT4 @ Pt
    nT4 /= nT4.norm()
    nT4 = nT4.permutate((2,), (0, 1, 3))
    return nT4
