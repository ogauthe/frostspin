import numpy as np
import scipy.linalg as lg

from svd_tools import svd_truncate, sparse_svd


def construct_projectors(R, Rt, chi):
    U, S, V, _ = svd_truncate(R @ Rt, chi)
    s12 = 1 / np.sqrt(S)  # S contains no 0
    # convention: projectors have shape (last_chi*D**2,chi)
    # since values of last_chi and D are not known (nor used) here
    #  00'      1
    #  ||       |
    #  Pt       P
    #  |        ||
    #  1        00'
    Pt = Rt @ V.conj().T * s12
    P = R.T @ U.conj() * s12
    return P, Pt


def construct_projectors_U1(corner1, corner2, corner3, corner4, chi, cutoff=1e-16):
    # once corner are constructed, no reshape or transpose is done. Decompose corner in
    # U(1) sectors as soon as they are constructed, then construct halves and R @ Rt
    # blockwise only. SVD and projectors can also be computed blockwise in same loop.
    # There are 4 sets of colors, since 2 adjacent blocks share one. Interior indices
    # refer to indices of each color blocks between R and Rt, where projectors are
    # needed.
    k = 0
    max_k = sum(min(chi, m.shape[0]) for m in corner1.blocks)  # upper bound
    P = np.zeros((corner2.shape[1], max_k))
    Pt = np.zeros((corner2.shape[1], max_k))
    S = np.empty(max_k)
    colors = np.empty(max_k, dtype=np.int8)
    shared = (
        set(corner1.block_colors)
        .intersection(corner2.block_colors)
        .intersection(corner3.block_colors)
        .intersection(corner4.block_colors)
    )
    for c in shared:
        m1, _, _ = corner1.get_block_row_col_with_color(c)
        m2, _, proj_indices = corner2.get_block_row_col_with_color(c)
        m3, _, _ = corner3.get_block_row_col_with_color(c)
        m4, _, _ = corner4.get_block_row_col_with_color(c)

        r = m1 @ m2
        rt = m3 @ m4
        m = r @ rt
        if min(m.shape) < 3 * chi:  # use full svd for small blocks
            u, s, v = lg.svd(m, full_matrices=False, overwrite_a=True)
        else:
            u, s, v = sparse_svd(m, k=chi, maxiter=1000)

        d = min(chi, s.size)  # may be smaller than chi
        S[k : k + d] = s[:d]
        colors[k : k + d] = c
        # not all blocks are used, the information which color sectors are used is
        # only known here. Need it to find row indices for P and Pt.
        P[proj_indices, k : k + d] = r.T @ u[:, :d].conj()
        Pt[proj_indices, k : k + d] = rt @ v[:d].T.conj()
        k += d

    s_sort = S[:k].argsort()[::-1]
    S = S[s_sort]
    cut = min(chi, (S > cutoff * S[0]).nonzero()[0][-1] + 1)
    s12 = 1 / np.sqrt(S[:cut])
    colors = colors[s_sort[:cut]]
    P = P[:, s_sort[:cut]] * s12
    Pt = Pt[:, s_sort[:cut]] * s12
    return P, Pt, colors


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
    nT = np.tensordot(
        T, P.reshape(A.shape[5], A.shape[5], T.shape[3], P.shape[1]), ((3,), (2,))
    )
    nT = nT.transpose(0, 3, 1, 4, 2, 5).copy()
    nT = np.tensordot(A, nT, ((4, 5), (0, 1)))
    nT = nT.transpose(0, 1, 4, 5, 2, 3, 6, 7).copy()
    nT = np.tensordot(A.conj(), nT, ((0, 1, 4, 5), (0, 1, 2, 3)))
    nT = (
        nT.transpose(3, 1, 4, 2, 0, 5)
        .copy()
        .reshape(Pt.shape[0], A.shape[2] ** 2 * P.shape[1])
    )
    nT = (Pt.T @ nT).reshape(Pt.shape[1], A.shape[2], A.shape[2], P.shape[1])
    nT /= nT.max()
    return nT


def renormalize_corner_P(C, T, P):
    """
    Renormalize corner C using projector P
    CPU: 2*chi**3*D**2
    """
    # use C3 as reference. At least one transpose is needed.

    #   0         0
    # 2-|       1-|
    # 3-|  -->  2-|
    #   1         3
    #             0
    #           1-|
    nC = T.transpose(0, 2, 3, 1).reshape(T.shape[0] * T.shape[2] ** 2, C.shape[0])
    nC = (nC @ C).reshape(T.shape[0], P.shape[0])
    #           0
    #       /-1-|
    # 1-P=01--2-|
    #       \   |
    #        \3-|
    nC = nC @ P
    nC /= nC.max()
    return nC


def renormalize_corner_Pt(C, T, Pt):
    """
    Renormalize corner C using projector Pt
    CPU: 2*chi**3*D**2
    """
    # use C4 as reference, no transpose needed

    #  0
    #  |-1
    #  |-2
    #  3
    #  0
    #  |-1
    nC = T.reshape(T.shape[0] * T.shape[1] ** 2, C.shape[0])
    nC = (nC @ C).reshape(T.shape[0], Pt.shape[0])
    #  0
    #  |-1-\
    #  |-2--10=P-1
    #  |   /
    #  |-3/
    nC = nC @ Pt
    nC /= nC.max()
    return nC


def renormalize_C1_up(C1, T4, P):
    """
    Renormalize corner C1 from an up move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C1.T, T4.transpose(3, 0, 1, 2), P).T


def renormalize_T1(Pt, T1, A, P):
    """
    Renormalize edge T1 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    nT1 = renormalize_T(Pt, T1.transpose(1, 2, 3, 0), A.transpose(0, 1, 4, 5, 2, 3), P)
    return nT1.swapaxes(0, 3)


def renormalize_C2_up(C2, T2, Pt):
    """
    Renormalize corner C2 from an up move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C2, T2.transpose(1, 2, 3, 0), Pt)


def renormalize_C2_right(C2, T1, P):
    """
    Renormalize corner C2 from right move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C2.T, T1.transpose(3, 0, 1, 2), P).T


def renormalize_T2(Pt, A, T2, P):
    """
    Renormalize edge T2 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    nT2 = renormalize_T(Pt, T2.transpose(2, 3, 0, 1), A.transpose(0, 1, 5, 2, 3, 4), P)
    return nT2.transpose(0, 3, 1, 2)


def renormalize_C3_right(C3, T3, Pt):
    """
    Renormalize corner C3 from right move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C3.T, T3.transpose(3, 0, 1, 2), Pt).T


def renormalize_C3_down(C3, T2, P):
    """
    Renormalize corner C3 from down move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C3, T2, P)


def renormalize_T3(Pt, T3, A, P):
    """
    Renormalize edge T3 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    return renormalize_T(Pt, T3, A, P).transpose(1, 2, 0, 3)


def renormalize_C4_down(C4, T4, Pt):
    """
    Renormalize corner C4 from a down move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C4, T4, Pt)


def renormalize_C4_left(C4, T3, P):
    """
    Renormalize corner C4 from a left move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C4.T, T3.transpose(2, 3, 0, 1), P).T


def renormalize_T4(Pt, T4, A, P):
    """
    Renormalize edge T4 using projectors P and Pt
    CPU: 2*chi**2*D**4*(a*d*D**2 + chi)
    """
    nT4 = renormalize_T(Pt, T4.transpose(1, 2, 3, 0), A.transpose(0, 1, 3, 4, 5, 2), P)
    return nT4.swapaxes(0, 3)


def renormalize_C1_left(C1, T1, Pt):
    """
    Renormalize corner C1 from a left move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C1, T1, Pt)
