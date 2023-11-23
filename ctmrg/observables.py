import numpy as np
import scipy.sparse.linalg as slg

# TODO: write class MPS, with unit cell and elementary tensors


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
    for u, d in zip(Tup_list[1:], Tdown_list[1:]):
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

    up_matrices = [np.ascontiguousarray(T.reshape(-1, T.shape[-1])) for T in Tup_list]
    down_matrices = [
        np.ascontiguousarray(T.swapaxes(-2, -1).reshape(-1, T.shape[-2]))
        for T in Tdown_list
    ]
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
        for m_u, m_d in zip(up_matrices, down_matrices):
            y = (m_u @ y).reshape(-1, m_d.shape[0]) @ m_d
        return y.ravel()

    tm = slg.LinearOperator((sh_vec[0] * sh_vec[1],) * 2, matvec=transfer_matvec)
    return tm


def mps_tranfer_open_leg(left_edge, up_mps, down_mps):
    r"""
    Apply MPS transfer matrix without contracting physical legs to obtain reduced
    density matrix

    Parameters
    ----------
    left_edge : ndarray
        Left MPS environment, typically left eigenvector of MPS transfer matrix. May
        include an additional middle leg (physical legs from former calls)
    up_mps : ndarray
        MPS up tensors, with or without ancilla leg
    down_mps : ndarray
        MPS down tensors, with or without ancilla leg

    Transfer matrix is defined as acting on a left edge state by adding MPS tensors
    from left to right.

    Notes
    -----
    Edge legs can be either
    -0            /0
    |    OR     1-
    -1            \2

    MPS legs may have or not have ancilla:
       2-Tup-0              3-Tup-0
          |                    ||
          1                    12
                      OR
          0                    01
          |                    ||
       2-Tdown-1           3-Tdown-2
    """
    if up_mps.ndim == 3:  # add ancila
        up = up_mps[:, :, None]
        down = np.ascontiguousarray(down_mps.transpose(2, 0, 1)[None])
    else:
        up = up_mps
        down = np.ascontiguousarray(down_mps.transpose(1, 3, 0, 2))
    if left_edge.ndim == 2:  # add middle leg
        new_edge = left_edge[:, None]
    else:
        new_edge = left_edge
    new_edge = np.tensordot(up, new_edge, ((3,), (0,)))
    new_edge = new_edge.transpose(0, 3, 1, 2, 4).copy()
    new_edge = np.tensordot(new_edge, down, ((3, 4), (0, 1)))
    new_edge = new_edge.reshape(up_mps.shape[0], -1, down_mps.shape[2])
    return new_edge


def compute_mps_rdm(up_mps_list, down_mps_list, ncell=1):
    """
    Expected leg ordering:

          2-up-0            3-up-0
             |                ||
             1                12
                      OR
             0                01
             |                ||
          2-down-1         3-down-2
    """
    d = up_mps_list[0].shape[1]
    D = up_mps_list[0].shape[-1]
    tm = construct_mps_transfer_matrix(up_mps_list, down_mps_list)
    (lval,), left_edge = slg.eigs(tm, k=1)
    if abs(lval.imag / lval.real) > 1e-6:
        raise ValueError("leading eigenvalue is not real")
    if np.linalg.norm(left_edge.imag) / np.linalg.norm(left_edge.real) > 1e-6:
        raise ValueError("left edge is not real")
    left_edge = np.ascontiguousarray(left_edge.real.reshape(D, 1, D))

    for i in range(ncell):
        for up, down in zip(up_mps_list, down_mps_list):
            left_edge = mps_tranfer_open_leg(left_edge, up, down)

    nt = len(up_mps_list) * ncell
    left_edge = left_edge.swapaxes(0, 1).reshape(d ** (2 * nt), D**2)
    tmT = construct_mps_transfer_matrix(up_mps_list, down_mps_list, transpose=True)
    (rval,), right_edge = slg.eigs(tmT, k=1)
    if abs(rval.imag / rval.real) > 1e-6:
        raise ValueError("leading eigenvalue is not real")
    if np.linalg.norm(right_edge.imag) / np.linalg.norm(right_edge.real) > 1e-6:
        raise ValueError("right edge is not real")
    if abs(1.0 - rval / lval) > 1e-6:
        raise ValueError("right and left eigenvalues differ")
    right_edge = right_edge.real

    rdm = left_edge @ right_edge
    perm = tuple(range(0, 2 * nt, 2)) + tuple(range(1, 2 * nt, 2))
    rdm = rdm.reshape((d,) * (2 * nt)).transpose(perm).reshape(d**nt, d**nt)
    rdm /= rdm.trace()
    return rdm


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
    except slg.ArpackNoConvergence as err:
        print("ARPACK did not converge", err)
        vals = err.eigenvalues
        print(f"Keep {vals.size} converged eigenvalues")
    except TypeError:  # crash when chi is too small, see scipy issue 16725
        print(f"WARNING: transfer matrix shape {op.shape} is too small for nval={nval}")
        return np.nan * np.ones((nval,))

    vals = vals[np.abs(vals).argsort()[::-1]]
    vals /= vals[0]

    if np.linalg.norm(vals.imag) < 1e-6:  # norm(vals.real) ~ 1
        vals = vals.real
    else:
        # complex eigenvalues mean wavector != (m*pi/Lx, n*pi/Ly)
        # this happens if correlation do not match unit cell size, which is always the
        # case for incommensurate correlations.
        print("Warning: transfer matrix eigenvalues are not real")
    return vals
