import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg

from misc_tools.svd_tools import find_chi_largest
from .asymmetric_tensor import AsymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor
from .o2_symmetric_tensor import O2_SymmetricTensor
from .su2_symmetric_tensor import SU2_SymmetricTensor

symmetric_tensor_types = {
    "trivial": AsymmetricTensor,
    "U1": U1_SymmetricTensor,
    "O2": O2_SymmetricTensor,
    "SU2": SU2_SymmetricTensor,
}


def get_symmetric_tensor_type(symmetry):
    """
    Get SymmetricTensor subclass implementing symmetry specified by 'symmetry'.

    Parameters
    ----------
    symmetry : str
        Symmetry group. Must match implemented symmetry. Currently implemented
        symmetries are 'trivial', 'U1' and 'SU2'.
    """
    try:
        st_type = symmetric_tensor_types[symmetry]
    except KeyError:
        raise ValueError(f"Unknown symmetry '{symmetry}'")
    return st_type


def symmetric_sparse_eigs(
    ST,
    reps,
    sig0,
    nvals,
    matmat,
    dmax_full=100,
    rng=None,
    maxiter=4000,
    tol=0,
    return_dense=True,
):
    """
    Find nvals eigenvalues for an implicit square matrix defined by matmat.

    Parameters
    ----------
    ST : type
        Symmetry tensor type of the implicit matrix M.
    reps : enumerable of representations
        Row representations for M
    sig0 : bool 1D array
        Signature for M rows
    nvals : int
        Number of eigenvalues to compute
    matmat : callable
        Apply matrix A to a symmetric tensor
    dmax_full : int
        Maximum block size to use dense eigvals.
    rng : numpy random generator
        Random number generator.
    maxiter : int
        Maxiter for Arpack.
    tol : float
        Arpack tol.
    return_dense : bool
        Whether to return a dense numpy array or a list of sector wise values. Default
        is True.

    Returns
    -------
    if return_dense:
        vals : 1D complex array
            Computed eigenvalues, as a dense array with multiplicites, sorted by
            decreasing absolute value. The last multiplet is cut to return exactly
            nvals values.
    else:
        vals : tuple of 1D complex array
            Eigenvalues for each non empty block, sorted by decreasing absolute value.
        block_irreps : int ndarray
            Irrep for each kept block
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(reps)
    sigm = np.empty((2 * n,), dtype=bool)
    sigm[:n] = sig0
    sigm[n:] = ~sig0
    block_irreps, block_shapes = ST.get_block_sizes(reps, reps, sigm)
    assert all(block_shapes[:, 0] == block_shapes[:, 1])

    nblocks = block_shapes.shape[0]
    sparse = []
    full = []
    blocks = []
    dims = np.empty((nblocks,), dtype=int)
    for bi in range(nblocks):
        irr = block_irreps[bi]
        dims[bi] = ST.irrep_dimension(irr)
        if block_shapes[bi, 0] > max(dmax_full, 2 * nvals / dims[bi]):
            sparse.append(bi)
        else:
            full.append(bi)
            blocks.append(np.eye(block_shapes[bi, 0]))

    sigv = np.ones((n + 1,), dtype=bool)
    sigv[:-1] = sig0
    ev_blocks = [None] * nblocks
    abs_ev_blocks = [None] * nblocks

    if full:
        irr_full = np.ascontiguousarray(block_irreps[full])
        rfull = ST.init_representation(block_shapes[full, 0], irr_full)
        full_blocks = ST(reps, (rfull,), blocks, irr_full, sigv)
        full_blocks = matmat(full_blocks)
        for i, bi in enumerate(full):
            ev = lg.eigvals(full_blocks.blocks[i])
            abs_ev = np.abs(ev)
            so = abs_ev.argsort()[::-1]
            ev_blocks[bi] = ev[so]
            abs_ev_blocks[bi] = abs_ev[so]

    for bi in sparse:
        irr = block_irreps[bi : bi + 1]
        brep = ST.init_representation(np.ones((1,), dtype=int), irr)
        sh = block_shapes[bi]

        def matvec(x):
            st = ST(reps, (brep,), (x[:, None],), irr, sigv)
            st = matmat(st)
            i = st.block_irreps.searchsorted(irr)[0]
            return st.blocks[i].ravel()

        op = slg.LinearOperator(sh, matvec=matvec)
        v0 = rng.random((sh[0],)) - 0.5
        k = nvals // dims[bi] + 1
        try:
            ev = slg.eigs(
                op, k=k, v0=v0, maxiter=maxiter, tol=tol, return_eigenvectors=False
            )
        except slg.ArpackNoConvergence as err:
            print("ARPACK did not converge", err)
            ev = err.eigenvalues
            print(f"Keep {ev.size} converged eigenvalues")

        abs_ev = np.abs(ev)
        so = abs_ev.argsort()[::-1]
        ev_blocks[bi] = ev[so]
        abs_ev_blocks[bi] = abs_ev[so]

    cuts = find_chi_largest(abs_ev_blocks, nvals, dims=dims)

    if return_dense:
        vals = np.empty((cuts @ dims,), dtype=np.complex128)
        k = 0
        for bi in range(nblocks):
            bv = ev_blocks[bi][: cuts[bi]]
            for d in range(dims[bi]):
                vals[k : k + cuts[bi]] = bv
                k += cuts[bi]

        so = np.argsort(np.abs(vals))[-1 : -nvals - 1 : -1]
        vals = np.ascontiguousarray(vals[so])
        return vals

    non_empty = cuts.nonzero()[0]
    block_irreps = np.ascontiguousarray(block_irreps[non_empty])
    final_ev = [None] * non_empty.size
    for i, bi in enumerate(non_empty):
        final_ev[i] = ev_blocks[bi][: cuts[bi]]
    return tuple(final_ev), block_irreps
