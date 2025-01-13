import numpy as np

import froSTspin


def compute_mps_transfer_spectrum(
    T1s,
    T3s,
    nvals,
    *,
    A_list=None,
    dmax_full=200,
    rng=None,
    maxiter=4000,
    tol=0,
):
    r"""
    Compute nval eigenvalues for the transfer matrix defined by the MPS in Tup_list and
    Tdown_list.

    Parameters
    ----------
    T1s : enum of SymmetricTensor
        Elementary cell MPS top tensors, from right to left. See leg ordering below.
    T3s : enum of SymmetricTensor
        Elementary cell MPS bottom tensors, from right to left.
    nvals : int
        Number of eigenvalues to compute.
    A_list : None or enum of SymmetricTensor
        Elementary cell site tensor, from right to left.
    dmax_full : int
        Maximum block size to use dense eigvals.
    rng : numpy random generator
        Random number generator. Used to initialize starting vectors for each block.
        If None, a new random generator is created with default_rng().
    maxiter : int
        Maximum number of Lanczos update iterations allowed in Arpack.
    tol : float
        Arpack tol.


    Notes
    -----
    Transfer matrix is defined as acting on a left edge state by adding MPS tensors
    from *right* to *left*.

    Expected leg ordering:

        3-Tup-0
           ||           0 2
           12            \|
                        5-A-3
                          |\
           01             4 1
           ||
       3-Tdown-2


    If A_list is not provided, the applied transfer matrix is made of 2 rows: one of Tup
    and one of Tdown.
           --T1--
             ||
           --T3--

    If A_list is provided, the applied transfer matrix is made of 3 rows: one of Tup,
    one of A and one of Tdown. This is much more expensive but should be more precise,
    especially for small chi.
           --T1--
             ||
           ==AA*=
             ||
           --T3--

    """
    ST = type(T1s[0])
    dtype = T1s[0].dtype

    if A_list is None:
        Ts_up = [T.permute((3, 1, 2), (0,)) for T in T1s]
        Ts_down = [T.permute((3,), (2, 0, 1)) for T in T3s]
        reps = (Ts_up[0].col_reps[0], Ts_down[0].col_reps[0])
        sig0 = np.empty((2,), dtype=bool)
        sig0[0] = ~Ts_up[0].signature[3]
        sig0[1] = ~Ts_down[0].signature[1]

        def matmat(st0):
            # bilayer matrix-vector product, MPS tensors going from right to left
            # (input is on the right).
            st = st0.permute((0,), (1, 2))
            for mu, md in zip(Ts_up, Ts_down, strict=True):
                st = mu @ st
                st = st.permute((3, 1, 2), (0, 4))
                st = md @ st
                st = st.permute((1,), (0, 2))
            return st.permute((0, 1), (2,))

    else:
        Ts_up = [T.permute((3, 1, 2), (0,)) for T in T1s]
        Ts_down = [T.permute((3,), (2, 0, 1)) for T in T3s]
        As = [A.permute((0, 1, 4, 5), (3, 2)) for A in A_list]
        Acs = [A.dagger().permute((2, 3), (4, 5, 1, 0)) for A in A_list]
        reps = (
            Ts_up[0].col_reps[0],
            As[0].col_reps[0],
            Acs[0].col_reps[2],
            Ts_down[0].col_reps[0],
        )
        sig0 = np.empty((4,), dtype=bool)
        sig0[0] = ~Ts_up[0].signature[3]
        sig0[1] = ~As[0].signature[4]
        sig0[2] = ~Acs[0].signature[4]
        sig0[3] = ~Ts_down[0].signature[1]

        def matmat(st0):
            st = st0.permute((0,), (1, 2, 3, 4))
            for mu, md, A, Aconj in zip(Ts_up, Ts_down, As, Acs, strict=True):
                st = mu @ st
                st = st.permute((3, 1), (0, 2, 4, 5, 6))
                st = A @ st
                st = st.permute((0, 1, 6, 5), (2, 3, 4, 7, 8))
                st = Aconj @ st
                st = st.permute((5, 2, 0), (1, 3, 4, 6))
                st = md @ st
                st = st.permute((3,), (2, 1, 0, 4))
            return st.permute((0, 1, 2, 3), (4,))

    try:
        vals = ST.eigs(
            matmat,
            nvals,
            reps=reps,
            signature=sig0,
            dtype=dtype,
            dmax_full=dmax_full,
            rng=rng,
            maxiter=maxiter,
            tol=tol,
            compute_vectors=False,
        )
        vals = vals.toarray()
    except Exception as err:  # not very stable
        print("WARNING: transfer matrix spectrum computation failed")
        print("Error:", err)
        return np.nan * np.ones((nvals,))

    sortperm = np.abs(vals).argsort()[: -nvals - 1 : -1]
    vals = vals[sortperm]
    vals /= vals[0]

    if np.linalg.norm(vals.imag) < 1e-6:  # norm(vals.real) ~ 1
        vals = vals.real
    elif not froSTspin.config["quiet"]:
        # complex eigenvalues mean wavector != (m*pi/Lx, n*pi/Ly)
        # this happens if correlation do not match unit cell size, which is always the
        # case for incommensurate correlations.
        print("Info: CTMRG transfer matrix eigenvalues are not real")
    return vals
