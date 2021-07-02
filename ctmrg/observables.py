import numpy as np
import scipy.sparse.linalg as slg


def construct_dense_transfer_matrix(Tup_list, Tdown_list):
    """
    Expensive contruction of the transfer matrix as a dense ndarray. Refer to
    get_mps_transfer_matrix for parameters and conventions.
    """
    if len(Tup_list) != len(Tdown_list):
        raise ValueError("len(Tup_list) != len(Tdown_list)")

    if Tup_list[0].ndim == 3:
        Tup_list = [T[:, None] for T in Tup_list]  # add empty axis to recover 4 legs
        Tdown_list = [T[None] for T in Tdown_list]

    u0 = Tup_list[0]
    d0 = Tdown_list[0]
    #  1-u0-0
    #     |
    #  3-d0-2
    tm = np.tensordot(u0, d0, ((1, 2), (0, 1)))
    #   /2-u0-0\
    # 1=    |   =0
    #   \3-d0-1/
    tm = tm.swapaxes(1, 2).reshape(u0.shape[0] * d0.shape[2], -1)
    for (u, d) in zip(Tup_list[1:], Tdown_list[1:]):
        #  1-u-0
        #    |
        #  3-d-2
        tm_i = np.tensordot(u, d, ((1, 2), (0, 1)))
        #    /2-u-0\
        #  1=   |   =0
        #    \3-d-1/
        tm_i = tm_i.swapaxes(1, 2).reshape(u.shape[0] * d.shape[2], -1)
        tm = tm_i @ tm
        del tm_i
    return tm


def construct_mps_transfer_matrix(Tup_list, Tdown_list, transpose=False):
    """
    Construct MPS transfer matrix as a scipy LinearOperator, with on the fly matrix
    product.

    Parameters
    ----------
    Tup_list : enum of ndarray
        Elementary cell MPS top tensors, from left to right. See leg ordering below.
    Tdown_list : enum of ndarray
        Elementary cell MPS bottom tensors, from left to right.
    transpose : bool, optional
        Whether to transpose the transfer matrix (can be used to obtain left
        eigenvectors). Tup_list and Tdown_list still need to be ordered from left to
        right in both cases. Default is False.

    Transfer matrix is defined as acting on a left edge state by adding MPS tensors
    from left to right. If transposed, it acts on a right edge state by adding tensors
    from right to left.

    Expected leg ordering:

       2-Tup-0              3-Tup-0
          |                    ||
          1                    12
                      OR
          0                    01
          |                    ||
       2-Tdown-1           3-Tdown-2

    That is there can be either one or two middle legs (physical + ancila, CTM double
    layer edge environment tensor).
    """
    if len(Tup_list) != len(Tdown_list):
        raise ValueError("len(Tup_list) != len(Tdown_list)")

    if transpose:
        Tup_list = [T.swapaxes(0, -1) for T in Tup_list[::-1]]
        Tdown_list = [T.swapaxes(-2, -1) for T in Tdown_list[::-1]]

    up_matrices = [T.reshape(-1, T.shape[-1]) for T in Tup_list]
    down_matrices = [T.swapaxes(-2, -1).reshape(-1, T.shape[-2]) for T in Tdown_list]
    sh_vec = (Tup_list[0].shape[-1], Tdown_list[0].shape[-1])

    def transfer_matvec(x):
        # bilayer matrix-vector product, MPS tensors going from left to right
        # (input is on the left). Transpose to act from right to left.
        #
        #  -0     2-Tup-0            ------Tup-0       -0
        #  |         |          =>   |      |    =>    |
        #  |         1               |      1          |
        #  |                         |                 |
        #  |         0               |      0          |
        #  |         |               |      |          |
        #  -1     1-Tdown-2          -2  1-Tdown-2     -1
        y = x.reshape(sh_vec)
        for (m_u, m_d) in zip(up_matrices, down_matrices):
            y = (m_u @ y).reshape(-1, m_d.shape[0]) @ m_d
        return y.ravel()

    tm = slg.LinearOperator((sh_vec[0] * sh_vec[1],) * 2, matvec=transfer_matvec)
    return tm


def compute_mps_transfer_spectrum(
    Tup_list, Tdown_list, nval, v0=None, ncv=None, maxiter=1000, tol=0
):
    """
    Compute nval eigenvalues for the transfer matrix defined by the MPS in Tup_list and
    Tdown_list.

    Parameters
    ----------
    Tup_list : enum of ndarray
        Elementary cell MPS top tensors, from left to right. See leg ordering below.
    Tdown_list : enum of ndarray
        Elementary cell MPS bottom tensors, from left to right.
    nval : int
        Number of eigenvalues to compute.
    The other parameters correspond to scipy.linalg.eigs optional parameters.

    Transfer matrix is defined as acting on a left edge state by adding MPS tensors
    from left to right. If transposed, it acts on a right edge state by adding tensors
    from right to left.

    Expected leg ordering:

       2-Tup-0              3-Tup-0
          |                    ||
          1                    12
                      OR
          0                    01
          |                    ||
       2-Tdown-1           3-Tdown-2

    That is there can be either one or two middle legs (physical + ancila, CTM double
    layer edge environment tensor).
    """
    op = construct_mps_transfer_matrix(Tup_list, Tdown_list)
    try:
        vals = slg.eigs(
            op,
            k=nval,
            v0=v0,
            ncv=ncv,
            maxiter=maxiter,
            tol=tol,
            return_eigenvectors=False,
        )
        if np.linalg.norm(vals.imag) / np.linalg.norm(vals.real) > 1e-6:
            print("Error: eigenvalues are not real")
            vals = np.nan * np.ones(nval)
        else:
            vals = vals.real[np.abs(vals).argsort()[::-1]]
            vals = vals / vals[0]
    except slg.ArpackNoConvergence as err:
        print("ARPACK did not converge", err)
        vals = np.nan * np.ones(nval)
    return vals
