import numpy as np
import scipy.sparse.linalg as slg


def get_transfer_matrix(Tup_list, Tdown_list):
    """
    Construct environment transfer matrix as a scipy LinearOperator, with on the fly
    matrix product.
    """
    assert len(Tup_list) == len(Tdown_list), "Tup_list != Tdown_list"
    sh_vec = (Tup_list[0].shape[3], Tdown_list[0].shape[3])
    sh_op = (sh_vec[0] * sh_vec[1],) * 2
    up_matrices = [T.reshape(-1, T.shape[3]) for T in Tup_list]
    down_matrices = [T.swapaxes(2, 3).reshape(-1, T.shape[2]) for T in Tdown_list]

    def transfer_mat_dot(x):
        y = x.reshape(sh_vec)
        for (m_u, m_d) in zip(up_matrices, down_matrices):
            y = (m_u @ y).reshape(-1, m_d.shape[0]) @ m_d
        return y.ravel()

    op = slg.LinearOperator(sh_op, matvec=transfer_mat_dot)
    return op


def compute_corr_length(Tup_list, Tdown_list, v0=None, ncv=None, maxiter=1000, tol=0):
    op = get_transfer_matrix(Tup_list, Tdown_list)
    try:
        vals = slg.eigs(
            op, k=2, v0=v0, ncv=ncv, maxiter=maxiter, tol=tol, return_eigenvectors=False
        )
        v2, v1 = np.sort(np.abs(vals))
        xi = len(Tup_list) / np.log(v1 / v2)
    except slg.ArpackNoConvergence as err:
        print("ARPACK did not converge", err)
        xi = np.nan
    return xi
