import numpy as np

from froSTspin.misc_tools.svd_tools import find_chi_largest, robust_svd, sparse_svd


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
    # P = R.transpose() @ U.conj() / s
    # Pt = Rt @ V.transpose().conj() / s

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
    r_blocks, rt_blocks, u_blocks, s_blocks, v_blocks = (
        [None] * n_blocks for i in range(5)
    )

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
    block_chi = np.minimum(chi + 10, block_chi)

    dims = []
    for bi in range(n_blocks):  # compute SVD on the fly
        r_blocks[bi] = corner1.blocks[ind1[bi]] @ corner2.blocks[ind2[bi]]
        rt_blocks[bi] = corner3.blocks[ind3[bi]] @ corner4.blocks[ind4[bi]]
        m = r_blocks[bi] @ rt_blocks[bi]
        dims.append(corner2.irrep_dimension(corner2.block_irreps[ind2[bi]]))
        if min(m.shape) < max(100, 6 * block_chi[bi]):  # use full svd for small blocks
            u, s, v = robust_svd(m)
        else:
            # a good precision is required for singular values, especially with pseudo
            # inverse. If precision is not good enough, reduced density matrix are less
            # hermitian. This requires a large number of computed vectors (ncv).
            ncv = int(ncv_ratio * block_chi[bi]) + 10
            u, s, v = sparse_svd(m, k=block_chi[bi], ncv=ncv, maxiter=1000)

        u_blocks[bi] = u
        s_blocks[bi] = s
        v_blocks[bi] = v

    # keep chi largest singular values + last multiplet
    block_cuts = find_chi_largest(
        s_blocks, chi, dims=dims, rcutoff=rcutoff, degen_ratio=degen_ratio
    )

    # second loop: construct projectors
    p_blocks = []
    pt_blocks = []
    non_empty = block_cuts.nonzero()[0]
    # construct P.transpose() blocks to avoid conjugating any representation
    for bi in non_empty:
        cut = block_cuts[bi]
        s12 = 1.0 / np.sqrt(s_blocks[bi][:cut])
        p_blocks.append(s12[:, None] * u_blocks[bi][:, :cut].T.conj() @ r_blocks[bi])
        pt_blocks.append(rt_blocks[bi] @ v_blocks[bi][:cut].T.conj() * s12)

    block_irreps = corner2.block_irreps[ind2[non_empty]]
    mid_rep = corner2.init_representation(block_cuts[non_empty], block_irreps)
    s2 = corner2.signature[corner2.n_row_reps :]
    sp = np.zeros((corner2.n_row_reps + 1), dtype=bool)
    sp[1:] = corner2.signature[corner2.n_row_reps :]
    spt = np.ones((corner2.n_row_reps + 1), dtype=bool)
    spt[:-1] = ~s2
    P = type(corner2)(
        (mid_rep,), corner2.col_reps, p_blocks, block_irreps, sp
    ).transpose()
    Pt = type(corner2)(corner2.col_reps, (mid_rep,), pt_blocks, block_irreps, spt)
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
    nT = T @ P.permute((2,), (0, 1, 3))
    nT = nT.permute((0, 3), (1, 4, 2, 5))
    nT = A @ nT
    nT = nT.permute((0, 1, 4, 5), (2, 3, 6, 7))
    nT = A.permute((0, 1, 4, 5), (2, 3)).dagger() @ nT
    nT = nT.permute((3, 1, 4), (2, 0, 5))
    nT = Pt.transpose() @ nT
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
    nC = nC.permute((0,), (1, 2, 3))
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
    nC = nC.permute((0,), (1, 2, 3))
    nC = nC @ Pt
    nC /= nC.norm()
    return nC


def renormalize_C1_up(C1, T4, P):
    """
    Renormalize corner C1 from an up move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(
        C1.transpose(), T4.permute((3, 1, 2), (0,)), P
    ).transpose()


def renormalize_T1(Pt, T1, A, P):
    """
    Renormalize edge T1 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    nT1 = renormalize_T(
        Pt, T1.permute((1, 2, 3), (0,)), A.permute((0, 1, 4, 5), (2, 3)), P
    )
    return nT1.permute((3,), (1, 2, 0))


def renormalize_C2_up(C2, T2, Pt):
    """
    Renormalize corner C2 from an up move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C2, T2.permute((1, 2, 3), (0,)), Pt)


def renormalize_C2_right(C2, T1, P):
    """
    Renormalize corner C2 from right move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(
        C2.transpose(), T1.permute((3, 1, 2), (0,)), P
    ).transpose()


def renormalize_T2(Pt, T2, A, P):
    """
    Renormalize edge T2 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    nT2 = renormalize_T(
        Pt, T2.permute((2, 3, 0), (1,)), A.permute((0, 1, 5, 2), (3, 4)), P
    )
    return nT2.permute((0,), (3, 1, 2))


def renormalize_C3_right(C3, T3, Pt):
    """
    Renormalize corner C3 from right move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(
        C3.transpose(), T3.permute((3, 0, 1), (2,)), Pt
    ).transpose()


def renormalize_C3_down(C3, T2, P):
    """
    Renormalize corner C3 from down move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(C3, T2.permute((0, 2, 3), (1,)), P)


def renormalize_T3(Pt, T3, A, P):
    """
    Renormalize edge T3 using projectors P and Pt
    CPU: 2*chi**2*D**4*(d*a*D**2 + chi)
    """
    nT3 = renormalize_T(
        Pt, T3.permute((0, 1, 2), (3,)), A.permute((0, 1, 2, 3), (4, 5)), P
    )
    return nT3.permute((1, 2, 0), (3,))


def renormalize_C4_down(C4, T4, Pt):
    """
    Renormalize corner C4 from a down move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C4, T4.permute((0, 1, 2), (3,)), Pt)


def renormalize_C4_left(C4, T3, P):
    """
    Renormalize corner C4 from a left move using projector P
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_P(
        C4.transpose(), T3.permute((2, 0, 1), (3,)), P
    ).transpose()


def renormalize_T4(Pt, T4, A, P):
    """
    Renormalize edge T4 using projectors P and Pt
    CPU: 2*chi**2*D**4*(a*d*D**2 + chi)
    """
    nT4 = renormalize_T(
        Pt, T4.permute((1, 2, 3), (0,)), A.permute((0, 1, 3, 4), (5, 2)), P
    )
    return nT4.permute((3,), (1, 2, 0))


def renormalize_C1_left(C1, T1, Pt):
    """
    Renormalize corner C1 from a left move using projector Pt
    CPU: 2*chi**3*D**2
    """
    return renormalize_corner_Pt(C1, T1.permute((0, 1, 2), (3,)), Pt)
