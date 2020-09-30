import numpy as np

from toolsU1 import default_color
from svd_tools import svd_truncate, sparse_svd


def construct_projectors(
    R, Rt, chi, ext_colors=default_color, int_colors=default_color
):
    if not ext_colors.size or not int_colors.size:  # no symmetry: bruteforce
        U, S, V, colors = svd_truncate(R @ Rt, chi)
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
        return P, Pt, colors

    # R @ Rt is only used for its SVD, which is done in U(1) blocks. Since python loop
    # over U(1) sectors is always done, better computing those blocks only instead of
    # full matrix product. Projectors can also be computed blockwise in same loop.
    # Since contraction can be done on the both sides of R and Rt, there are only 2 sets
    # of colors: inner (along contraction) and extern, simpler algo
    ext_sort = ext_colors.argsort()
    sorted_ext_colors = ext_colors[ext_sort]
    ext_blocks = np.array(
        [
            0,
            *((sorted_ext_colors[:-1] != sorted_ext_colors[1:]).nonzero()[0] + 1),
            R.shape[0],
        ]
    )
    int_sort = int_colors.argsort()
    sorted_int_colors = int_colors[int_sort]
    int_blocks = [
        0,
        *((sorted_int_colors[:-1] != sorted_int_colors[1:]).nonzero()[0] + 1),
        R.shape[0],
    ]
    k = 0
    max_k = np.minimum(chi, ext_blocks[1:] - ext_blocks[:-1]).sum()
    P = np.zeros((R.shape[1], max_k))
    Pt = np.zeros((R.shape[1], max_k))
    S = np.empty(max_k)
    colors = np.empty(max_k, dtype=np.int8)
    k, eb, ib, ebmax, ibmax = 0, 0, 0, ext_blocks.size - 1, len(int_blocks) - 1
    while eb < ebmax and ib < ibmax:
        if sorted_ext_colors[ext_blocks[eb]] == sorted_int_colors[int_blocks[ib]]:
            ext_indices = ext_sort[ext_blocks[eb] : ext_blocks[eb + 1]]
            int_indices = int_sort[int_blocks[ib] : int_blocks[ib + 1]]

            r = np.ascontiguousarray(R[ext_indices[:, None], int_indices])
            rt = np.ascontiguousarray(Rt[int_indices[:, None], ext_indices])
            m = r @ rt
            if min(m.shape) < 3 * chi:  # use full svd for small blocks
                u, s, v = np.linalg.svd(m, full_matrices=False)
            else:
                u, s, v = sparse_svd(m, k=chi, maxiter=1000)

            d = min(chi, s.size)  # may be smaller than chi
            S[k : k + d] = s[:d]
            colors[k : k + d] = sorted_ext_colors[ext_blocks[eb]]
            P[int_indices, k : k + d] = r.T @ u[:, :d].conj()
            Pt[int_indices, k : k + d] = rt @ v[:d].T.conj()
            k += d
            eb += 1
            ib += 1
        elif sorted_ext_colors[ext_blocks[eb]] < sorted_int_colors[int_blocks[ib]]:
            eb += 1
        else:
            ib += 1

    s_sort = S[:k].argsort()[::-1]  # if some interior color does not exit, k<max_k
    S = S[s_sort]
    cut = min(chi, (S > 0).nonzero()[0][-1] + 1)
    s12 = np.sqrt(S[:cut])
    colors = colors[s_sort[:cut]]
    P = P[:, s_sort[:cut]] / s12
    Pt = Pt[:, s_sort[:cut]] / s12
    return P, Pt, colors


###############################################################################
# conventions: renormalize_C(C,T,P)
#              renormalize_T(Pt,T,A,P)
###############################################################################


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
    nT1 = np.tensordot(
        T1, Pt.reshape(T1.shape[3], A.shape[5], A.shape[5], Pt.shape[1]), ((3,), (0,))
    )
    nT1 = np.tensordot(nT1, A, ((1, 3), (2, 5)))
    nT1 = np.tensordot(nT1, A.conj(), ((1, 2, 4, 5), (2, 5, 0, 1)))
    nT1 = nT1.transpose(1, 3, 5, 0, 2, 4).reshape(
        Pt.shape[1] * A.shape[4] ** 2, P.shape[0]
    )
    nT1 = (nT1 @ P).reshape(Pt.shape[1], A.shape[4], A.shape[4], P.shape[1])
    nT1 = nT1.swapaxes(0, 3).copy()
    nT1 /= np.amax(nT1)
    return nT1


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
    nT2 = np.tensordot(
        T2, Pt.reshape(T2.shape[0], A.shape[2], A.shape[2], Pt.shape[1]), ((0,), (0,))
    )
    nT2 = np.tensordot(nT2, A, ((3, 1), (2, 3)))
    nT2 = np.tensordot(nT2, A.conj(), ((4, 5, 2, 1), (0, 1, 2, 3)))
    nT2 = nT2.transpose(1, 3, 5, 0, 2, 4).reshape(
        Pt.shape[1] * A.shape[5] ** 2, P.shape[0]
    )
    nT2 = np.dot(nT2, P).reshape(Pt.shape[1], A.shape[5], A.shape[5], P.shape[1])
    nT2 = nT2.transpose(0, 3, 1, 2).copy()
    nT2 /= np.amax(nT2)
    return nT2


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
    nT3 = np.tensordot(
        T3, P.reshape(T3.shape[3], A.shape[5], A.shape[5], P.shape[1]), ((3,), (0,))
    )
    nT3 = np.tensordot(A, nT3, ((4, 5), (0, 3)))
    nT3 = np.tensordot(nT3, A.conj(), ((0, 1, 4, 6), (0, 1, 4, 5)))
    nT3 = nT3.transpose(0, 4, 3, 2, 1, 5).reshape(
        A.shape[2] ** 2 * P.shape[1], Pt.shape[0]
    )
    nT3 = np.dot(nT3, Pt).reshape(A.shape[2], A.shape[2], P.shape[1], Pt.shape[1])
    nT3 = nT3.swapaxes(2, 3).copy()
    nT3 /= np.amax(nT3)
    return nT3


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
    nT4 = np.tensordot(
        P.reshape(T4.shape[0], A.shape[2], A.shape[2], P.shape[1]), T4, ((0,), (0,))
    )
    nT4 = np.tensordot(nT4, A, ((0, 3), (2, 5)))
    nT4 = np.tensordot(nT4, A.conj(), ((0, 2, 4, 5), (2, 5, 0, 1)))
    nT4 = nT4.transpose(0, 2, 4, 1, 3, 5).reshape(
        P.shape[1] * A.shape[3] ** 2, Pt.shape[0]
    )
    nT4 = np.dot(nT4, Pt).reshape(P.shape[1], A.shape[3], A.shape[3], Pt.shape[1])
    nT4 /= np.amax(nT4)
    return nT4


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
