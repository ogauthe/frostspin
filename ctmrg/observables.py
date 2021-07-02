import numpy as np
import scipy.sparse.linalg as slg


def get_mps_transfer_matrix(Tup_list, Tdown_list=None):
    """
    Construct MPS transfer matrix as a scipy LinearOperator, with on the fly matrix
    product.

    Parameters
    ----------
    Tup_list: enum of ndarray
        Elementary cell MPS top tensors, from left to right. See leg ordering below.
    Tdown_list: None or enum of ndarray
        Elementary cell MPS bottom tensors, from left to right. If not provided,
        they are assumed to be the same as Tup, up to transposition.

    Expected leg ordering:

       2-Tup-0              3-Tup-0
          |                    ||
          1                    12
                      OR
          0                    01
          |                    ||
       2-Tdown-1           3-Tdown-2

    That is there can be either one or two middle legs (physical + ancila, CTM double
    layer edge environment tensor)
    """
    if Tdown_list is None:
        if Tup_list[0].ndim == 3:
            perm = (1, 2, 0)
        else:
            perm = (1, 2, 3, 0)
        Tdown_list = [T.transpose(perm) for T in Tup_list]
    elif len(Tup_list) != len(Tdown_list):
        raise ValueError("len(Tup_list) != len(Tdown_list)")

    up_matrices = [T.reshape(-1, T.shape[-1]) for T in Tup_list]
    down_matrices = [T.swapaxes(-1, -2).reshape(-1, T.shape[-2]) for T in Tdown_list]
    sh_vec = (Tup_list[0].shape[3], Tdown_list[0].shape[3])
    sh_op = (sh_vec[0] * sh_vec[1],) * 2

    def transfer_mat_dot(x):
        # bilayer matrix-vector product, MPS tensors going from left to right
        # (input is at the left)
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

    op = slg.LinearOperator(sh_op, matvec=transfer_mat_dot)
    return op


def compute_transfer_spectrum(
    Tup_list, Tdown_list, nval, v0=None, ncv=None, maxiter=1000, tol=0
):
    op = get_mps_transfer_matrix(Tup_list, Tdown_list)
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
        if np.linalg.norm(vals.imag / vals.real) > 1e-6:
            print("Error: eigenvalues are not real")
            vals = np.nan * np.ones(nval)
        else:
            vals = vals.real[np.abs(vals).argsort()[::-1]]
            vals = vals / vals[0]
    except slg.ArpackNoConvergence as err:
        print("ARPACK did not converge", err)
        vals = np.nan * np.ones(nval)
    return vals
