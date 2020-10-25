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
    #         0'5-A-30'
    #        /    |\   \
    #     1-P     4 1   Pt-1
    #        \    01    /
    #         \   ||   /
    #          03-T3-20
    nT = np.tensordot(
        T, P.reshape(T.shape[3], A.shape[5], A.shape[5], P.shape[1]), ((3,), (0,))
    )
    nT = nT.transpose(0, 3, 1, 4, 2, 5).copy()
    nT = np.tensordot(A, nT, ((4, 5), (0, 1)))
    nT = nT.transpose(0, 1, 4, 5, 2, 3, 6, 7).copy()
    nT = np.tensordot(A.conj(), nT, ((0, 1, 4, 5), (0, 1, 2, 3)))
    nT = (
        nT.transpose(4, 3, 1, 2, 0, 5)
        .copy()
        .reshape(Pt.shape[0], A.shape[2] ** 2 * P.shape[1])
    )
    nT = (Pt.T @ nT).reshape(Pt.shape[1], A.shape[2], A.shape[2], P.shape[1])
    nT /= np.amax(nT)
    return nT


def renormalize_C1_up(C1, T4, P):
    """
    Renormalize corner C1 from an up move using projector P
    CPU: 2*chi**3*D**2
    """
    #  C1-0
    #  |
    #  1
    #  0
    #  |
    #  T4=1,2
    #  |
    #  3
    nC1 = np.tensordot(C1, T4, ((1,), (0,)))
    #  C1-0---\
    #  |       0
    #  T4=1,2-/
    #  |
    #  3 -> 1
    nC1 = nC1.reshape(P.shape[0], T4.shape[3])
    #  C1-\
    #  |   00=P-1 ->0
    #  T4-/
    #  |
    #  1
    nC1 = P.T @ nC1
    nC1 /= np.amax(nC1)
    return nC1


def renormalize_T1(Pt, T1, A, P):
    """
    Renormalize edge T1 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    #      0  3-T1-0  0
    #     /     ||      \
    #    /    0 12       \
    # 1-Pt     \ 2        P-1
    #    \      \|       /
    #     \0'  5-A-3  0'/
    #            |\
    #            4 1

    # P and Pt come as matrices, useful for construction
    # here gives messy reshapes, no effect on renlormalize_C
    return (
        renormalize_T(Pt, T1.transpose(1, 2, 3, 0), A.transpose(0, 1, 4, 5, 2, 3), P)
        .swapaxes(0, 3)
        .copy()
    )


def renormalize_C2_up(C2, T2, Pt):
    """
    Renormalize corner C2 from an up move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #    0<-1-C2
    #          |
    #          0
    #          0
    #          |
    #     2,3=T2
    #          |
    #          1
    nC2 = np.tensordot(C2, T2, ((0,), (0,)))
    #         10-C2
    #        /    |
    # 1-Pt=01     |
    #        \22=T2
    #          3  |
    #             1 ->0
    nC2 = nC2.swapaxes(0, 1).reshape(T2.shape[1], Pt.shape[0])
    nC2 = nC2 @ Pt
    nC2 /= np.amax(nC2)
    return nC2


def renormalize_C2_right(C2, T1, P):
    """
    Renormalize corner C2 from right move using projector P
    CPU: 2*chi**3*D**2
    """
    #   1<- 3-T1-01-C2
    #         ||     |
    #         12     0
    #          \   /
    #            0
    nC2 = np.tensordot(C2, T1, ((1,), (0))).reshape(P.shape[0], T1.shape[3])
    #      1-T1-C2
    #         \/
    #          0
    #          0
    #          ||
    #          P
    #          |
    #          1
    nC2 = P.T @ nC2
    nC2 /= np.amax(C2)
    return nC2


def renormalize_T2(Pt, A, T2, P):
    """
    Renormalize edge T2 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    #       1
    #       |
    #       Pt
    #     /   \
    #    0'    0
    #  0 2     0
    #   \|     |
    #  5-A-32-T2
    #    |\ 3/ |
    #    4 1   1
    #    0'    0
    #     \   /
    #       P
    #       |
    #       1
    return (
        renormalize_T(Pt, T2.transpose(2, 3, 0, 1), A.transpose(0, 1, 5, 2, 3, 4), P)
        .transpose(0, 3, 1, 2)
        .copy()
    )


def renormalize_C3_right(C3, T3, Pt):
    """
    Renormalize corner C3 from right move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #       01     0
    #       ||     |
    #     3-T3-21-C3
    nC3 = np.tensordot(C3, T3, ((1,), (2,)))
    #          1 ->0
    #          |
    #          Pt
    #          ||
    #          0
    #          0
    #         / \
    #        12  0
    #        ||  |
    #  1<- 3-T3-C3
    nC3 = Pt.T @ nC3.reshape(Pt.shape[0], T3.shape[3])
    nC3 /= np.amax(nC3)
    return nC3


def renormalize_C3_down(C3, T2, P):
    """
    Renormalize corner C3 from down move using projector P
    CPU: 2*chi**3*D**2
    """
    #         0
    #         |
    #     12=T2
    #     23  |
    #         1
    #         0
    #         |
    #     31-C3
    nC3 = np.tensordot(T2, C3, ((1,), (0,)))
    #          0
    #          |
    #  1-  21-T2
    #   \  32/ |
    #          |
    #      13-C3
    nC3 = nC3.transpose(0, 3, 1, 2).reshape(T2.shape[0], P.shape[0])
    #          0
    #          |
    #        /-T2
    #  1-P=01  |
    #        \-C3
    nC3 = nC3 @ P
    nC3 /= np.amax(nC3)
    return nC3


def renormalize_T3(Pt, T3, A, P):
    """
    Renormalize edge T3 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    #           0 2
    #            \|
    #         0'5-A-30'
    #        /    |\   \
    #     1-P     4 1   Pt-1
    #        \    01    /
    #         \   ||   /
    #          03-T3-20
    return renormalize_T(Pt, T3, A, P).transpose(1, 2, 0, 3).copy()


def renormalize_C4_down(C4, T4, Pt):
    """
    Renormalize corner C4 from a down move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #   0
    #   |
    #   T4=1,2
    #   |
    #   3
    #   0
    #   |
    #   C4-1 ->3
    nC4 = np.tensordot(T4, C4, ((3,), (0,)))
    #   0
    #   |  12
    #   T4=23  \
    #   |       10=Pt-1
    #   C4-31  /
    nC4 = nC4.transpose(0, 3, 1, 2).reshape(T4.shape[0], Pt.shape[0])
    nC4 = nC4 @ Pt
    nC4 /= np.amax(nC4)
    return nC4


def renormalize_C4_left(C4, T3, P):
    """
    Renormalize corner C4 from a left move using projector P
    CPU: 2*chi**3*D**2
    """
    #           12
    #     0     01
    #     |     ||
    #     C4-13-T3-2 ->3
    nC4 = np.tensordot(C4, T3, ((1,), (3,)))
    #        1 ->0
    #        |
    #        P
    #       ||
    #        0
    #        0
    #       /  \
    #     0     12
    #     |     ||
    #     C4----T3-3 -> 1
    nC4 = nC4.reshape(P.shape[0], T3.shape[2])
    nC4 = P.T @ nC4
    nC4 /= np.amax(nC4)
    return nC4


def renormalize_T4(Pt, T4, A, P):
    """
    Renormalize edge T4 using projectors P and Pt
    CPU: 2*chi**2*D**4*(a*d*D**2 + chi)
    """
    #       1
    #       |
    #       P
    #     /   \
    #    0     0'
    #    0   1 2
    #    |    \|
    #    T4=15-A-3
    #    |  2  |\
    #    3     4 0
    #    0     0'
    #     \   /
    #       Pt
    #       |
    #       1
    return (
        renormalize_T(Pt, T4.transpose(1, 2, 3, 0), A.transpose(0, 1, 3, 4, 5, 2), P)
        .swapaxes(0, 3)
        .copy()
    )


def renormalize_C1_left(C1, T1, Pt):
    """
    Renormalize corner C1 from a left move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #  C1-03-T1-0
    #  |     ||
    #  1     12
    #  3     12
    #  1     23
    #   \   /
    #     1
    nC1 = np.tensordot(T1, C1, ((3,), (0,)))
    nC1 = nC1.transpose(0, 3, 1, 2).reshape(T1.shape[0], Pt.shape[0])
    #  C1--T1-0
    #  |   |
    #   \ /
    #    1
    #    0
    #    ||
    #    Pt
    #    |
    #    1
    nC1 = nC1 @ Pt
    nC1 /= np.amax(nC1)
    return nC1
