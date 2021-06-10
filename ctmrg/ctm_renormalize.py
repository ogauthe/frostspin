import numpy as np
import scipy.linalg as lg

from misc_tools.svd_tools import svd_truncate, sparse_svd
from ctmrg.ctm_contract import add_a_blockU1


def construct_projectors(R, Rt, chi, cutoff, degen_ratio, window):
    U, S, V, _ = svd_truncate(
        R @ Rt, chi, cutoff=cutoff, degen_ratio=degen_ratio, window=window
    )
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


def construct_projectors_U1(
    corner1, corner2, corner3, corner4, chi, cutoff, degen_ratio, window
):
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
    for c in shared:  # avoid svd_truncate to compute SVD on the fly
        m1 = corner1.blocks[corner1.get_color_index(c)]
        i2 = corner2.get_color_index(c)
        m2 = corner2.blocks[i2]
        proj_indices = corner2.col_indices[i2]
        m3 = corner3.blocks[corner3.get_color_index(c)]
        m4 = corner4.blocks[corner4.get_color_index(c)]

        r = m1 @ m2
        rt = m3 @ m4
        m = r @ rt
        if min(m.shape) < 3 * chi:  # use full svd for small blocks
            try:
                u, s, v = lg.svd(m, full_matrices=False, overwrite_a=True)
            except lg.LinAlgError as err:
                print("Error in scipy dense SVD:", err)
                m = r @ rt  # overwrite_a=True may have erased it
                u, s, v = lg.svd(
                    m,
                    full_matrices=False,
                    overwrite_a=True,
                    check_finite=False,
                    driver="gesvd",
                )
        else:
            # for U(1) as SU(2) subgroup, no degen inside a color block
            u, s, v = sparse_svd(m, k=chi + window, maxiter=1000)

        d = min(chi, s.size)  # may be smaller than chi
        S[k : k + d] = s[:d]
        colors[k : k + d] = c
        # not all blocks are used, the information which color sectors are used is
        # only known here. Need it to find row indices for P and Pt.
        P[proj_indices, k : k + d] = r.T @ u[:, :d].conj()
        Pt[proj_indices, k : k + d] = rt @ v[:d].T.conj()
        k += d

    # TODO: benchmark with heapq.nlargest(chi, range(k), lambda i: S[i])
    s_sort = S[:k].argsort()[::-1]  # remove non-written values before sorting
    S = S[s_sort]
    cut = min(chi, (S > cutoff * S[0]).nonzero()[0][-1] + 1)
    # cut between multiplets, defined as S[i+1]/S[i] >= degen_ratio (>= for ratio=1)
    nnz = (S[cut:] <= degen_ratio * S[cut - 1 : -1]).nonzero()[0]
    if nnz.size:
        cut += nnz[0]
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


###############################################################################
# U(1) symmetric renormalize_T
###############################################################################


def renormalize_T1_U1(Pt, T1, a_ul, P, col_T1_r, col_Pt, col_a_ul, col_a_r, col_a_d):
    """
    Renormalize edge T1 using projectors P and Pt with U(1) symmetry
    CPU: highly depends on symmetry, worst case chi**2*D**8
    """
    # Pt -> left, need swapaxes
    # T1 -> up, transpose due to add_a_blockU1 conventions
    left = Pt.reshape(-1, T1.shape[3], Pt.shape[1]).swapaxes(0, 1).copy()
    nT1 = T1.transpose(1, 2, 0, 3).reshape(T1.shape[1] ** 2, T1.shape[0], T1.shape[3])
    nT1 = add_a_blockU1(nT1, left, a_ul, col_T1_r, col_Pt, col_a_ul, col_a_r, col_a_d)
    #             -T1-0'
    #            / ||
    #       1'-Pt==AA=0
    #            \ ||
    #               1'
    nT1 = P.T @ nT1
    nT1 /= nT1.max()
    dim_d = round(float(np.sqrt(nT1.shape[1] // Pt.shape[1])))
    nT1 = nT1.reshape(P.shape[1], dim_d, dim_d, Pt.shape[1])
    return nT1


def renormalize_T2_U1(Pt, T2, a_ur, P, col_T2_d, col_Pt, col_a_ur, col_a_d, col_a_l):
    """
    Renormalize edge T2 using projectors P and Pt with U(1) symmetry
    CPU: highly depends on symmetry, worst case chi**2*D**8
    """
    # Pt -> left, need swapaxes
    # T2 -> up
    left = Pt.reshape(-1, T2.shape[0], Pt.shape[1]).swapaxes(0, 1).copy()
    nT2 = T2.transpose(2, 3, 1, 0).reshape(T2.shape[2] ** 2, T2.shape[1], T2.shape[0])
    nT2 = add_a_blockU1(nT2, left, a_ur, col_T2_d, col_Pt, col_a_ur, col_a_d, col_a_l)
    #                 3
    #                 |
    #                Pt
    #              //  |
    #           2==AA=T2
    #              \\  |
    #               0  1
    nT2 = P.T @ nT2
    nT2 /= nT2.max()
    dim_d = round(float(np.sqrt(nT2.shape[1] // Pt.shape[1])))
    nT2 = nT2.reshape(P.shape[1], dim_d, dim_d, Pt.shape[1]).transpose(3, 0, 1, 2)
    return nT2


def renormalize_T3_U1(Pt, T3, a_dl, P, col_T3_r, col_P, col_a_dl, col_a_u, col_a_r):
    """
    Renormalize edge T3 using projectors P and Pt with U(1) symmetry
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
    left = P.reshape(-1, T3.shape[3], P.shape[1]).swapaxes(0, 1).copy()
    nT3 = T3.reshape(T3.shape[0] ** 2, T3.shape[2], T3.shape[3])
    nT3 = add_a_blockU1(nT3, left, a_dl, col_T3_r, col_P, col_a_dl, col_a_r, col_a_u)
    #               2
    #              ||
    #            //AA=0
    #         3-P  ||
    #            \-T3-1
    nT3 = Pt.T @ nT3
    nT3 /= nT3.max()
    dim_d = round(float(np.sqrt(nT3.shape[1] // P.shape[1])))
    nT3 = nT3.reshape(Pt.shape[1], dim_d, dim_d, P.shape[1]).transpose(1, 2, 0, 3)
    return nT3


def renormalize_T4_U1(Pt, T4, a_ul, P, col_T4_d, col_P, col_a_ul, col_a_r, col_a_d):
    """
    Renormalize edge T4 using projectors P and Pt with U(1) symmetry
    CPU: highly depends on symmetry, worst case chi**2*D**8
    """
    # we can use either a_ul or a_dl. In both cases, the second projector must be added
    # with nT4 = nT4 @ P / Pt due to a leg ordering. Use a_ul to have down leg in 3.
    # P -> up
    # T4 -> left
    nT4 = P.reshape(-1, T4.shape[0], P.shape[1]).swapaxes(1, 2).copy()
    left = T4.reshape(T4.shape[0], T4.shape[1] ** 2, T4.shape[3])
    nT4 = add_a_blockU1(nT4, left, a_ul, col_P, col_T4_d, col_a_ul, col_a_r, col_a_d)
    #              1
    #              |
    #              P
    #            / ||
    #          T4==AA=0
    #            \ ||
    #            3  2
    nT4 = nT4 @ Pt
    nT4 /= nT4.max()
    dim_d = round(float(np.sqrt(nT4.shape[0] // P.shape[1])))
    nT4 = nT4.reshape(dim_d, dim_d, P.shape[1], Pt.shape[1]).transpose(2, 0, 1, 3)
    return nT4
