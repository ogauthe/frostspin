import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg

from misc_tools.svd_tools import numba_find_chi_largest
from ctmrg.ctm_contract import add_a_bilayer


def construct_projectors(
    corner1,
    corner2,
    corner3,
    corner4,
    chi,
    block_chi_ratio,
    ncv_ratio,
    rcutoff,
    degen_ratio,
    last_renormalized,
):
    """
    Parameters
    ----------
    corner1 : SymmetricTensor
        First corner, either obtained by C-T//T-A contraction or just C (no absorb)
    corner2 : SymmetricTensor
        Second corner to contract
    corner3 : SymmetricTensor
        Third corner to contract
    corner4 : SymmetricTensor
        Fourth corner to contract
    chi : int
        Total number of singular values to keep
    block_chi_ratio : float
        For each symmetry block, compute block_chi = block_chi_ratio * last_block_chi
        singular values, where last_block_chi is the size of the symmetry sector during
        last truncation. Final number of kept singular values is ajusted according to
        chi.
    ncv_ratio : float
        For each symmetry sector, generate ncv = ncv_ratio * chi_block Lanczos vectors.
    degen_ratio : float
        ratio to consider values as degenerate (see numba_find_chi_largest
        documentation)
    last_renormalized : SymmetricTensor
        Last renormalized corner, used to estimate block sizes in SVD.
    """
    # factorize loops on different symmetry sectors, construct only blocks that will
    # appear in final projectors. Compute SVD blockwise on the fly for R @ Rt, without
    # storing all the blocks together.
    # This is a bit verbose, just an optimization of
    # R = corner1 @ corner2
    # Rt = corner3 @ corner4
    # M = R @ Rt
    # U, s, V = truncated_svd(M)
    # P = R.T @ U.conj() / s
    # Pt = Rt @ V.T.conj() / s
    assert (last_renormalized.col_reps[0] == corner2.col_reps[-1]).all()

    shared = sorted(
        set(corner1.block_irreps)
        .intersection(corner2.block_irreps)
        .intersection(corner3.block_irreps)
        .intersection(corner4.block_irreps)
    )
    shared = np.array(shared)
    n_blocks = shared.size
    ind1 = corner1.block_irreps.searchsorted(shared)
    ind2 = corner2.block_irreps.searchsorted(shared)
    ind3 = corner3.block_irreps.searchsorted(shared)
    ind4 = corner4.block_irreps.searchsorted(shared)

    # first loop: compute SVD for all blocks
    r_blocks, rt_blocks = [[None] * n_blocks for i in range(2)]
    u_blocks, s_blocks, v_blocks = [[None] * n_blocks for i in range(3)]

    # CTMRG is a fixed point algorithm: expect symmetry sectors to converge very fast.
    # Hence no need to consider worst case where all leading singular belong to the same
    # symmetry sector: in each block, compute the same number of values as were kept in
    # last iteration + some margin to fluctuate, as specified by block_chi_ratio
    block_chi = np.zeros(shared.shape, dtype=int)
    last_irreps = last_renormalized.block_irreps
    for bi, ind in enumerate(last_irreps.searchsorted(shared)):
        if ind < last_irreps.size and shared[bi] == last_irreps[ind]:
            block_chi[bi] = last_renormalized.blocks[ind].shape[1]
    block_chi = np.maximum(block_chi + 10, (block_chi_ratio * block_chi).astype(int))
    block_chi = np.minimum(chi, block_chi)

    for bi in range(n_blocks):  # SVD only for shared blocks
        c1 = corner1.blocks[ind1[bi]]
        c2 = corner2.blocks[ind2[bi]]
        c3 = corner3.blocks[ind3[bi]]
        c4 = corner4.blocks[ind4[bi]]
        if max(max(c.shape for c in (c1, c2, c3, c4))) < max(100, 6 * block_chi[bi]):
            m = c1 @ c2 @ c3 @ c4  # full matrix product and svd for small blocks
            u, s, v = lg.svd(m, full_matrices=False, overwrite_a=True)
        else:  # never construct R, Rt and R @ Rt for large blocks
            c1H, c2H, c3H, c4H = c1.conj().T, c2.conj().T, c3.conj().T, c4.conj().T
            n = c1.shape[0]

            def corner_XHX(x):
                return c4H @ (c3H @ (c2H @ (c1H @ (c1 @ (c2 @ (c3 @ (c4 @ x)))))))

            op = slg.LinearOperator(matvec=corner_XHX, shape=(n, n), dtype=c1.dtype)
            # a good precision is required for singular values, especially with pseudo
            # inverse. If precision is not good enough, reduced density matrix are less
            # hermitian. This requires a large number of computed vectors (ncv).
            ncv = int(ncv_ratio * block_chi[bi])
            eigvals, eigvec = slg.eigsh(op, k=block_chi[bi], ncv=ncv, maxiter=1000)
            u = c1 @ (c2 @ (c3 @ (c4 @ eigvec)))
            u, s, v = lg.svd(u, full_matrices=False, overwrite_a=True)
            v = v @ eigvec.T.conj()

        u_blocks[bi] = u
        s_blocks[bi] = s
        v_blocks[bi] = v

    # keep chi largest singular values + last multiplet
    s_blocks = tuple(s_blocks)
    block_cuts = numba_find_chi_largest(s_blocks, chi, rcutoff, degen_ratio)

    # second loop: construct projectors
    p_blocks = []
    pt_blocks = []
    non_empty = block_cuts.nonzero()[0]
    # construct P.T blocks to avoid conjugating any representation
    # never construct large matrices R and Rt, contract with truncated matrix first
    for bi in non_empty:
        cut = block_cuts[bi]
        s12 = 1.0 / np.sqrt(s_blocks[bi][:cut])
        p = s12[:, None] * u_blocks[bi][:, :cut].T.conj()
        p = p @ corner1.blocks[ind1[bi]] @ corner2.blocks[ind2[bi]]
        pt = v_blocks[bi][:cut].T.conj() * s12
        pt = corner3.blocks[ind3[bi]] @ (corner4.blocks[ind4[bi]] @ pt)
        p_blocks.append(p)
        pt_blocks.append(pt)

    block_irreps = corner2.block_irreps[ind2[non_empty]]
    mid_rep = corner2.init_representation(block_cuts[non_empty], block_irreps)
    P = type(corner2)((mid_rep,), corner2.col_reps, p_blocks, block_irreps).T
    Pt = type(corner2)(corner2.col_reps, (mid_rep,), pt_blocks, block_irreps)
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
    nT = T @ P.permutate((2,), (0, 1, 3))
    nT = nT.permutate((0, 3), (1, 4, 2, 5))
    nT = A @ nT
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
        Pt, T1.permutate((1, 2, 3), (0,)), A.permutate((0, 1, 4, 5), (2, 3)), P
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
        Pt, T2.permutate((2, 3, 0), (1,)), A.permutate((0, 1, 5, 2), (3, 4)), P
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
    nT3 = renormalize_T(
        Pt, T3.permutate((0, 1, 2), (3,)), A.permutate((0, 1, 2, 3), (4, 5)), P
    )
    return nT3.permutate((1, 2, 0), (3,))


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
        Pt, T4.permutate((1, 2, 3), (0,)), A.permutate((0, 1, 3, 4), (5, 2)), P
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
