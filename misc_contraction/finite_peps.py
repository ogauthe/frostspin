import numpy as np

import ctmrg.observables


def rdm_finite_PEPS_2x2(A, B, C, D):
    r"""
    Construct reduced density matrix on plaquette AB//CD with periodic boundary
    conditions. Tensors must follow standard leg ordering paurdl:
        0 2
         \|
        5-A-3
         /|
        1 4
    """
    #  0 2 6 4
    #   \| |/
    #  *-A-B-*
    #   /| |\
    #  1 3 7 5
    ab = np.tensordot(A, B, ((3, 5), (5, 3)))
    cd = np.tensordot(C, D, ((3, 5), (5, 3)))

    #  0 * * 2
    #   \| |/
    #  *-A-B-*
    #   /| |\
    #  1 | | 3
    #  4 | | 6
    #   \| |/
    #  *-C-D-*
    #   /| |\
    #  5 * * 7
    abcd = np.tensordot(ab, cd, ((2, 3, 6, 7), (3, 2, 7, 6)))
    del ab, cd
    abcd = np.tensordot(abcd, abcd, ((1, 3, 5, 7), (1, 3, 5, 7)))
    d4 = A.shape[0] * B.shape[0] * C.shape[0] * D.shape[0]
    abcd = abcd.reshape(d4, d4)
    abcd /= abcd.trace()
    return abcd


def mps_vert_2sites(A, B):
    #  0 *
    #   \|
    #  3-A-2
    #   /|
    #  1 |
    #  4 |
    #   \|
    #  7-B-6
    #   /|
    #  5 *
    mps = np.tensordot(A, B, ((2, 4), (4, 2)))
    #   3-mps-0
    #      ||
    #      12
    mps = mps.transpose(2, 6, 0, 4, 1, 5, 3, 7).reshape(
        A.shape[3] * B.shape[3],
        A.shape[0] * B.shape[0],
        A.shape[1] * B.shape[1],
        A.shape[5] * B.shape[5],
    )
    return mps


def contract_cylinder_2rows(A, B, C, D):
    r"""
    0 2
     \|
    5-A-3
     /|
    1 4
    """
    mps_AC = mps_vert_2sites(A, C)
    mps_BD = mps_vert_2sites(B, D)
    up_mps_list = [mps_AC, mps_BD]
    down_mps_list = [mps_AC.transpose(1, 2, 0, 3), mps_BD.transpose(1, 2, 0, 3)]

    rdm = ctmrg.observables.compute_mps_rdm(up_mps_list, down_mps_list)
    return rdm
