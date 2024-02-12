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
    dtype=None,
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
        Number of eigenvalues to compute.
    matmat : callable
        Apply matrix A to a symmetric tensor
    dtype : type
        Scalar type. Default is np.complex128.
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
            nvals (dense) values.
    else:
        vals : tuple of 1D complex array
            Eigenvalues for each non empty block, sorted by decreasing absolute value.
        block_irreps : int ndarray
            Irrep for each kept block
    """
    # 1) set parameters
    if dtype is None:
        dtype = np.complex128
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(reps)
    sigm = np.empty((2 * n,), dtype=bool)
    sigm[:n] = sig0
    sigm[n:] = ~sig0
    block_irreps, block_shapes = ST.get_block_sizes(reps, reps, sigm)
    nblocks = block_shapes.shape[0]
    assert all(block_shapes[:, 0] == block_shapes[:, 1])
    sigv = np.ones((n + 1,), dtype=bool)
    sigv[:-1] = sig0
    ev_blocks = [None] * nblocks
    abs_ev_blocks = [None] * nblocks

    # 2) split matrix blocks between full and sparse
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
            blocks.append(np.eye(block_shapes[bi, 0], dtype=dtype))

    # 3) construct full matrix blocks and call dense eig on them
    # use just one call of matmat on identity blocks to produce all blocks
    if full:
        irr_full = np.ascontiguousarray(block_irreps[full])
        rfull = ST.init_representation(block_shapes[full, 0], irr_full)
        st0 = ST(reps, (rfull,), blocks, irr_full, sigv)
        st1 = matmat(st0)
        for bi in full:
            irr = block_irreps[bi]
            bj = st1.block_irreps.searchsorted(irr)
            if bj < st1.nblocks and st1.block_irreps[bj] == irr:
                ev = lg.eigvals(st1.blocks[bj])
                abs_ev = np.abs(ev)
                so = abs_ev.argsort()[::-1]
                ev_blocks[bi] = ev[so]
                abs_ev_blocks[bi] = abs_ev[so]
            else:  # missing block means eigval = 0
                ev_blocks[bi] = np.zeros((block_shapes[bi, 0],), dtype=dtype)
                abs_ev_blocks[bi] = np.zeros((block_shapes[bi, 0],))

    # 4) for each sparse block, apply matmat to a SymmetricTensor with 1 block
    for bi in sparse:
        irr = block_irreps[bi]
        block_irreps_bi = block_irreps[bi : bi + 1]
        brep = ST.init_representation(np.ones((1,), dtype=int), block_irreps_bi)
        sh = block_shapes[bi]

        v0 = rng.normal(size=(sh[0],)).astype(dtype, copy=False)
        st0 = ST(reps, (brep,), (v0[:, None],), block_irreps_bi, sigv)
        st1 = matmat(st0)
        bj = st1.block_irreps.searchsorted(irr)

        # check that irr block actually appears in output
        if bj < st1.nblocks and st1.block_irreps[bj] == irr:

            def matvec(x):
                st0.blocks[0][:, 0] = x
                st1 = matmat(st0)
                y = st1.blocks[bj].ravel()  # implicitely assume bj does not depend on x
                return y

            op = slg.LinearOperator(sh, matvec=matvec, dtype=dtype)
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

        else:  # missing block
            ev_blocks[bi] = np.zeros((nvals,), dtype=dtype)
            abs_ev_blocks[bi] = np.zeros((nvals,))

    cuts = find_chi_largest(abs_ev_blocks, nvals, dims=dims)

    if return_dense:
        vals = np.empty((cuts @ dims,), dtype=dtype)
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
