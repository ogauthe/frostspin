import numpy as np

import symmetric_tensor.tools

# TODO: write class MPS, with unit cell and elementary tensors


def compute_mps_transfer_spectrum(
    Tup_list,
    Tdown_list,
    nvals,
    dmax_full=200,
    rng=None,
    maxiter=4000,
    tol=0,
):
    """
    Compute nval eigenvalues for the transfer matrix defined by the MPS in Tup_list and
    Tdown_list.

    Parameters
    ----------
    Tup_list : enum of SymmetricTensor
        Elementary cell MPS top tensors, from left to right. See leg ordering below.
    Tdown_list : enum of SymmetricTensor
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

    Tup_list = [T.permute((3, 1, 2), (0,)) for T in Tup_list]
    Tdown_list = [T.permute((3,), (2, 0, 1)) for T in Tdown_list]
    reps = (Tup_list[0].col_reps[0], Tdown_list[0].col_reps[0])
    sig0 = np.empty((2,), dtype=bool)
    sig0[0] = ~Tup_list[0].signature[3]
    sig0[1] = ~Tdown_list[0].signature[1]

    def matmat(st):
        # bilayer matrix-vector product, MPS tensors going from right to left
        # (input is on the right).
        st = st.permute((0,), (1, 2))
        for mu, md in zip(Tup_list, Tdown_list):
            st = mu @ st
            st = st.permute((3, 1, 2), (0, 4))
            st = md @ st
            st = st.permute((1,), (0, 2))
        st = st.permute((0, 1), (2,))
        return st

    try:
        vals = symmetric_tensor.tools.symmetric_sparse_eigs(
            type(Tup_list[0]),
            reps,
            sig0,
            nvals,
            matmat,
            dtype=Tup_list[0].dtype,
            dmax_full=dmax_full,
            rng=rng,
            maxiter=maxiter,
            tol=tol,
            return_dense=True,
        )
    except Exception as err:  # not very stable
        print("WARNING: transfer matrix spectrum computation failed")
        print("Error:", err)
        return np.nan * np.ones((nvals,))

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
