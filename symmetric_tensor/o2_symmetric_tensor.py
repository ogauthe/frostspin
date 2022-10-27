import numpy as np
import scipy.sparse as sp
import scipy.linalg as lg
import numba

from misc_tools.sparse_tools import custom_sparse_product, custom_double_sparse_product
from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor
from .u1_symmetric_tensor import U1_SymmetricTensor

_tol = 2e-13  # used only in assert


@numba.njit
def _numba_combine_O2(*reps):
    degen, irreps = reps[0]
    for r in reps[1:]:
        degen, irreps = _numba_elementary_combine_O2(degen, irreps, r[0], r[1])
    return np.concatenate((degen, irreps)).reshape(2, -1)


@numba.njit
def _numba_elementary_combine_O2(degen1, irreps1, degen2, irreps2):
    # 0o x 0o = 0e
    # 0e x 0e = 0e
    # 0e x 0o = 0o
    # 0o x 0e = 0o
    # 0o x n = n
    # 0e x n = n
    # n x n = 0o + 0e + 2n
    # n x m = |n-m| + (n+m)
    nmax = max(irreps1[-1], 0) + max(irreps2[-1], 0)
    degen = np.zeros((nmax + 2,), dtype=np.int64)

    for (d1, irr1) in zip(degen1, irreps1):
        for (d2, irr2) in zip(degen2, irreps2):
            d = d1 * d2
            if irr1 < 1:
                if irr2 < 1:
                    degen[(irr1 + irr2 + 1) % 2] += d
                else:
                    degen[irr2 + 1] += d
            else:
                if irr2 < 1:
                    degen[irr1 + 1] += d
                elif irr1 == irr2:
                    degen[0] += d
                    degen[1] += d
                    degen[2 * irr1 + 1] += d
                else:
                    degen[irr1 + irr2 + 1] += d
                    degen[abs(irr1 - irr2) + 1] += d

    irreps = np.arange(-1, nmax + 1)
    nnz = degen.nonzero()[0]
    return degen[nnz], irreps[nnz]


@numba.njit
def _numba_O2_representation_dimension(r):
    return (r[0] * ((r[1] > 0).astype(np.int64) + 1)).sum()


@numba.njit
def _numba_O2_rep_to_U1(r):
    """
    Map O(2) representation to U(1) representation
    O(2) representations starts with irreps 0odd, 0even which map to U(1) 0
    then O(2) irreps merge +n and -n U(1) for n>0
    maps to U(1) irrep +n and -n, with contiguous +n and -n sectors

    ex: [[1, 2, 1, 2], [-1, 0, 1, 2]] is mapped to [0, 0, 0, 1, -1, 2, 2, -2, -2]
    """
    # ordering is a bit more complex than consecutive pairs (n,-n)
    # hope to improve perf with these consecutive U(1) sectors
    ru1 = np.empty((_numba_O2_representation_dimension(r),), dtype=np.int8)
    i = 0
    k = 0
    if r[1, 0] == -1:
        ru1[: r[0, 0]] = 0
        k += r[0, 0]
        i += 1
    if r[1, i] == 0:
        ru1[k : k + r[0, i]] = 0
        k += r[0, i]
        i += 1
    for j in range(i, r.shape[1]):
        ru1[k : k + r[0, j]] = r[1, j]
        k += r[0, j]
        ru1[k : k + r[0, j]] = -r[1, j]
        k += r[0, j]
    return ru1


@numba.njit
def _numba_get_reflection_perm_sign(rep):
    """
    Construct basis permutation mapping vectors into their reflected form.

    Parameters
    ----------
    rep: O(2) representation
    """
    d = _numba_O2_representation_dimension(rep)
    perm = np.arange(d, dtype=np.uint64)
    sign = np.ones((d,), dtype=np.int8)
    i1 = np.searchsorted(rep[1], 1)  # slice containing 0o and 0e if they exist
    if rep[1, 0] == -1:
        sign[: rep[0, 0]] = -1
    k = rep[0, :i1].sum()
    for i in range(i1, rep.shape[1]):
        degen = rep[0, i]
        perm[k : k + degen] = np.arange(k + degen, k + 2 * degen, dtype=np.uint64)
        perm[k + degen : k + 2 * degen] = np.arange(k, k + degen, dtype=np.uint64)
        if rep[1, i] % 2:  # -1 sign for Sz<0 to Sz>0
            sign[k + degen : k + 2 * degen] = -1
        k += 2 * degen
    return perm, sign


@numba.njit(parallel=True)
def _numba_get_swapped(states, o2_reps):
    """
    Construct array of reflected states.

    states: (n,) uint64 array
        Dense indices to reflect, mono index form.
    o2_reps: tuple of nx O(2) representations
        Representation on each axis.
    """
    # avoid constructing array of multi-index states, which may be large.
    nx = len(o2_reps)
    perm_axes = []
    sign_axes = []
    for j in range(nx):
        p, s = _numba_get_reflection_perm_sign(o2_reps[j])
        perm_axes.append(p)
        sign_axes.append(s)

    sh = [_numba_O2_representation_dimension(r) for r in o2_reps]  # numba issue 8497
    sh = np.array(sh, dtype=np.uint64)
    strides = np.array([np.uint64(1), *sh[-1:0:-1]]).cumprod()[::-1].copy()
    n = states.shape[0]
    refl_states = np.empty((n,), dtype=np.uint64)  # Sz-reversed mapped indices
    signs = np.empty((n,), dtype=np.int8)
    for i in numba.prange(n):
        refl_s = 0
        s_sign = 1
        for j in range(nx):
            ind = states[i] // strides[j] % sh[j]
            refl_s += perm_axes[j][ind] * strides[j]
            s_sign *= sign_axes[j][ind]
        refl_states[i] = refl_s
        signs[i] = s_sign

    return refl_states, signs


@numba.njit(parallel=True)
def _numba_b0_arrays(sz0_states, o2_reps, nfxe_ar, nfxo):
    # 1st run parallel loop to get Sz-reflected states
    # make it the simplest possible, do not write arrays
    # we need to store refl_states and signs for fixed points anyway
    sz0_states_reversed, signs = _numba_get_swapped(sz0_states, o2_reps)

    # define types and sizes. dtype information is passed as nfxe_ar dtype
    dt = nfxe_ar.dtype
    nfxe = nfxe_ar[()]
    n_sz0 = sz0_states.size
    n_doublets = (n_sz0 - nfxe - nfxo) // 2
    ncoeff_even = nfxe + 2 * n_doublets  # number of coefficients in even projector
    ncoeff_odd = nfxo + 2 * n_doublets

    # initialize array sizes
    notfx = np.empty((n_doublets,), dtype=dt)
    erows = np.empty((ncoeff_even,), dtype=dt)
    erows[: nfxe + n_doublets] = np.arange(nfxe + n_doublets, dtype=dt)
    ecols = np.empty((ncoeff_even,), dtype=dt)
    ecoeff = np.empty((ncoeff_even,))
    ecoeff[:nfxe] = 1.0  # fixed points
    orows = np.empty((ncoeff_odd,), dtype=dt)
    orows[: nfxo + n_doublets] = np.arange(nfxo + n_doublets, dtype=dt)
    ocols = np.empty((ncoeff_odd,), dtype=dt)
    ocoeff = np.empty((ncoeff_odd,))
    ocoeff[:nfxo] = 1.0  # fixed points

    # fixed points require non-parallel loop to be detected
    ie, io, j = 0, 0, 0
    for i in range(n_sz0):
        if sz0_states[i] < sz0_states_reversed[i]:
            notfx[j] = i
            j += 1
        elif sz0_states[i] == sz0_states_reversed[i]:  # fixed point
            if signs[i] > 0:  # even
                ecols[ie] = i
                ie += 1
            else:  # odd
                ocols[io] = i
                io += 1

    # doublets are independent from each other once notfx is defined: parallel loop
    for i in numba.prange(n_doublets):
        j = np.searchsorted(sz0_states, sz0_states_reversed[notfx[i]])
        erows[nfxe + n_doublets + i] = nfxe + i
        ecols[nfxe + i] = notfx[i]
        ecols[nfxe + n_doublets + i] = j
        ecoeff[nfxe + i] = 1.0 / np.sqrt(2)
        ecoeff[nfxe + n_doublets + i] = signs[j] / np.sqrt(2)

        orows[nfxo + n_doublets + i] = nfxo + i
        ocols[nfxo + i] = notfx[i]
        ocols[nfxo + n_doublets + i] = j
        ocoeff[nfxo + i] = 1.0 / np.sqrt(2)
        ocoeff[nfxo + n_doublets + i] = -signs[j] / np.sqrt(2)
    return ecoeff, erows, ecols, ocoeff, orows, ocols


def _get_b0_projectors(o2_reps, sz_values):
    """
    Construct 0odd and 0even projectors. They allow to decompose U(1) Sz=0 sector into
    0odd (aka -1) and 0even (aka 0) sectors.

    Parameters
    ----------
    o2_reps : tuple of nx O(2) representations
        Representation on each axis.
    sz_valuses : (k,) int8 ndarray
        Sz values corresponding to merged o2_reps

    Returns
    -------
        po : coo_matrix
            Projector from Sz=0 block to 0odd sector
        pe : coo_matrix
            Projector from Sz=0 block to 0even sector
    """
    # sz_values can be recovered from o2_reps, but it is usually already constructed
    # when calling _get_b0_projectors. Also avoids to add leg signatures as input.

    # Some Sz=0 states are fixed points, mapped to the same state, either even or odd.
    # They belong to even or odd sector depending on signs and need to be sent in this
    # sector with coeff 1.
    # All the other states are doublets which requires even and odd combination to
    # produce 1 even and 1 odd states.
    # Hence the number of even (odd) states is number of even (odd) fixed points plus
    # the number of doublets.

    # To get simpler projectors, we first detect the fixed points and set them as the
    # first lines. Then doublets are considered.

    # compute number of odd and even fixed points from O(2) reps: they are product of 0
    nfxo = 0
    nfxe = 0
    temp = tuple(np.ascontiguousarray(r[:, : r[1].searchsorted(1)]) for r in o2_reps)
    if all(r.size for r in temp):  # may crash for empty rep
        fx_oe = O2_SymmetricTensor.combine_representations(temp, None)
        if fx_oe.shape[1] == 2:
            nfxo = fx_oe[0, 0]  # number of odd fixed points
            nfxe = fx_oe[0, 1]  # number of even fixed point
        elif fx_oe.shape[1] == 1:
            if fx_oe[1, 0] == -1:  # no even
                nfxo = fx_oe[0, 0]
            else:  # no odd
                nfxe = fx_oe[0, 0]

    sz0_states = (sz_values == 0).nonzero()[0].view(np.uint64)
    n_sz0 = sz0_states.size
    # scipy issue 16774: default index dtype is int32 with force cast from int64
    dt = np.int32 if n_sz0 < 2**31 else np.int64
    nfxe_ar = np.array(nfxe, dtype=dt)  # store dtype information for numba

    ecoeff, erows, ecols, ocoeff, orows, ocols = _numba_b0_arrays(
        sz0_states, tuple(o2_reps), nfxe_ar, nfxo
    )
    n_doublets = (n_sz0 - nfxe - nfxo) // 2
    pe = sp.coo_matrix((ecoeff, (erows, ecols)), shape=(nfxe + n_doublets, n_sz0))
    po = sp.coo_matrix((ocoeff, (orows, ocols)), shape=(nfxo + n_doublets, n_sz0))
    return po, pe


@numba.njit(parallel=True)
def _numba_generate_block(rso, cso, b0, rsign, csign):
    b = np.empty((rso.size, cso.size), dtype=b0.dtype)
    for i in numba.prange(rso.size):
        for j in numba.prange(cso.size):
            b[rso[i], cso[j]] = rsign[i] * csign[j] * b0[i, j]
    return b


@numba.njit(parallel=True)
def _numba_O2_transpose(
    old_shape,
    old_blocks,
    old_block_sz,
    old_row_sz,
    old_col_sz,
    axes,
    new_block_sz,
    new_row_sz,
    new_col_sz,
    rmaps,
    rsigns,
    cmaps,
    csigns,
):
    """
    Construct new irrep blocks after permutation.

    Parameters
    ----------
    old_shape : (ndim,) uint64 array
        Tensor shape before transpose.
    old_blocks : homogeneous tuple of onb C-array
        Blocks before transpose, with 0e and 0o combined in one Sz=0 block. Sz<0 blocks
        should not be included.
    old_block_sz : (onb,) int8 ndarray
        Block Sz values before transpose.
    old_row_sz : (old_nrows,) int8 ndarray
        Row Sz values before transpose.
    old_col_sz : (old_ncols,) int8 ndarray
        Column Sz values before transpose.
    axes : tuple of ndim integers
        Axes permutation.
    new_block_sz: 1D int8 array
        Sz label for each block, with only Sz >=0.
    new_row_sz : (new_nrows,) int8 ndarray
        Row Sz values after transpose.
    new_col_sz : (new_ncols,) int8 ndarray
        Column Sz values after transpose.
    rmaps: tuple of old_nrr uint64 1D array
        Index mapping to Sz-reflected state for each row axis.
    rsigns: tuple of old_nrr int8 1D array
        Sign to add when reflecting for each row axis.
    cmaps: tuple of ndim - old_nrr uint64 1D array
        Index mapping to Sz-reflected state for each column axis.
    csigns: tuple of ndim - old_nrr int8 1D array
        Sign to add when reflecting for each column axis.

    Returns
    -------
    blocks : tuple of nnb C-array
        Blocks after transpose, with Sz=0 block instead of 0e/0o.
    block_sz : (nnb,) int8 ndarray
        Block Sz values after transpose.

    Note that old_shape is a ndarray and not a tuple.
    old_blocks MUST be homogeneous tuple of C-array, using F-array sometimes
    fails in a non-deterministic way.
    """
    ###################################################################################
    # very similar to U(1)
    ###################################################################################
    # 1) construct strides before and after transpose for rows and cols
    old_nrr = len(rmaps)
    old_ncr = len(cmaps)
    ndim = old_nrr + old_ncr
    rstrides1 = np.ones((old_nrr,), dtype=np.uint64)
    rstrides1[1:] = old_shape[old_nrr - 1 : 0 : -1]
    rstrides1 = rstrides1.cumprod()[::-1].copy()
    rmod = old_shape[:old_nrr]

    cstrides1 = np.ones((old_ncr,), dtype=np.uint64)
    cstrides1[1:] = old_shape[-1:old_nrr:-1]
    cstrides1 = cstrides1.cumprod()[::-1].copy()
    cmod = old_shape[old_nrr:]

    new_strides = np.ones(ndim, dtype=np.uint64)
    for i in range(ndim - 1, 0, -1):
        new_strides[axes[i - 1]] = new_strides[axes[i]] * old_shape[axes[i]]
    rstrides2 = new_strides[:old_nrr]
    cstrides2 = new_strides[old_nrr:]

    # 2) find unique Sz>=0 in rows and relate them to blocks and indices.
    n = len(new_block_sz)
    block_rows = np.empty((new_row_sz.size,), dtype=np.uint64)
    row_irrep_count = np.zeros((n,), dtype=np.uint64)
    new_row_block_indices = np.empty(new_row_sz.shape, dtype=np.uint64)
    for i in range(new_row_sz.size):
        for j in range(n):
            if new_row_sz[i] == new_block_sz[j]:
                block_rows[i] = row_irrep_count[j]
                row_irrep_count[j] += 1
                new_row_block_indices[i] = j
                break

    # 3) find each column index inside new blocks
    ncs = np.uint64(new_col_sz.size)
    block_cols = np.empty((ncs,), dtype=np.uint64)
    col_irrep_count = np.zeros((n,), dtype=np.uint64)
    for i in range(ncs):
        for j in range(n):
            if new_col_sz[i] == new_block_sz[j]:
                block_cols[i] = col_irrep_count[j]
                col_irrep_count[j] += 1
                break

    # 4) initialize block sizes. We know blocks are non empty thanks to block_sz.
    dt = old_blocks[0].dtype
    new_blocks = [
        np.zeros((row_irrep_count[i], col_irrep_count[i]), dtype=dt) for i in range(n)
    ]

    # 5) copy all coeff from all blocks to new destination
    # use simpler O(2) mapping sz -sz?
    # then map is i -> i + (sz[i] > 0) * 2 - bool(sz[i])
    # still cannot be vectorized due to sz[i] call
    for bi in range(old_block_sz.size):
        # nonzero returns int64, indexing is faster with unsigned int
        ori = (old_row_sz == old_block_sz[bi]).nonzero()[0].view(np.uint64)
        block_nrows = ori.size
        rszb_mat = np.empty((block_nrows,), dtype=np.uint64)
        rsign = np.empty((block_nrows,), dtype=np.int8)
        for i in numba.prange(block_nrows):
            permuted_s = 0
            refl_s = 0
            s_sign = 1
            for j in range(old_nrr):
                s = ori[i] // rstrides1[j] % rmod[j]
                permuted_s += s * rstrides2[j]
                refl_s += rmaps[j][s] * rstrides2[j]
                s_sign *= rsigns[j][s]
            ori[i] = permuted_s  # overwrite ori with permuted state
            rszb_mat[i] = refl_s
            rsign[i] = s_sign

        oci = (old_col_sz == old_block_sz[bi]).nonzero()[0].view(np.uint64)
        block_ncols = oci.size
        cszb_mat = np.empty((block_ncols,), dtype=np.uint64)
        csign = np.empty((block_ncols,), dtype=np.int8)
        for i in numba.prange(block_ncols):
            permuted_s = 0
            refl_s = 0
            s_sign = 1
            for j in range(old_ncr):
                s = oci[i] // cstrides1[j] % cmod[j]
                permuted_s += s * cstrides2[j]
                refl_s += cmaps[j][s] * cstrides2[j]
                s_sign *= csigns[j][s]
            oci[i] = permuted_s
            cszb_mat[i] = refl_s
            csign[i] = s_sign

        for i in numba.prange(block_nrows):
            for j in numba.prange(block_ncols):
                nr, nc = divmod(ori[i] + oci[j], ncs)

                # if new Sz >= 0, move coefficient, same as U(1)
                if new_row_sz[nr] >= 0:
                    new_bi = new_row_block_indices[nr]
                    nri = block_rows[nr]
                    nci = block_cols[nc]
                    new_blocks[new_bi][nri, nci] = old_blocks[bi][i, j]

                # if new Sz <= 0, move old Sz-reversed coeff
                if old_block_sz[bi] and new_row_sz[nr] <= 0:  # reflect
                    nrb, ncb = divmod(rszb_mat[i] + cszb_mat[j], ncs)
                    new_bi = new_row_block_indices[nrb]
                    nri = block_rows[nrb]
                    nci = block_cols[ncb]
                    s = rsign[i] * csign[j]
                    new_blocks[new_bi][nri, nci] = s * old_blocks[bi][i, j]

    return new_blocks


class O2_SymmetricTensor(NonAbelianSymmetricTensor):
    """
    SymmetricTensor with global O(2) symmetry. Implement it as semi direct product of
    Z_2 and U(1). Irreps of U(1) are labelled by integer n. For n>0 even, one gets a
    dimension 2 irrep of O(2) by coupling n and -n. For n odd, this is a projective
    representation. The sector n=0 is split into 2 1D irreps, 0 even and 0 odd.

    irrep -1 labels 0 odd
    irrep 0 labels 0 even
    irrep n>0 labels irrep (+n, -n)
    """

    # impose consecutive +n and -n sectors
    # impose, for n even, same sign in +n and -n (differs from SU(2))
    # impose, for n odd, sign from +n to -n (differs from SU(2))

    # there are 2 possibilites to store reprensetation info
    # => mimic abelian, with format
    # [1, -1, 0, 1]
    # [o,  e, o, e]
    # but then where to store mapping on Sz-reversed?
    # could be a third row of rep, but then rep has to have int64 dtype => heavy

    # OR full non-abelian: rep = array([degen, irreps])
    # with irreps being -1 for 0odd, 0 for 0even, n for +/-n, which need to be taken
    # consecutive AND impose some rules like n even => even, n odd => sign appears from
    # Sz<0 to Sz>0
    # may have difficulties when transposing / taking adjoint
    # last possibility: third layer with sign
    # still impose consecutive n / -n, but allows for more flexibility in signs

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    @classmethod
    @property
    def symmetry(cls):
        return "O2"

    @classmethod
    @property
    def singlet(cls):
        return np.array([[1], [0]])

    @staticmethod
    def representation_dimension(rep):
        return _numba_O2_representation_dimension(rep)

    @staticmethod
    def irrep_dimension(irrep):
        return 1 + (irrep > 0)

    @staticmethod
    def combine_representations(reps, signature):
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_O2(*reps)
        return reps[0]

    @staticmethod
    def conjugate_representation(rep):
        return rep

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        # not fully efficient: 1st construct U(1) symmetric tensor to get abelian blocks
        # then split sector 0 into 0even and 0odd
        # discard sectors with n < 0
        # keep sectors with n > 0 as they are
        u1_row_reps = tuple(_numba_O2_rep_to_U1(r) for r in row_reps)
        u1_col_reps = tuple(_numba_O2_rep_to_U1(r) for r in col_reps)
        tu1 = U1_SymmetricTensor.from_array(arr, u1_row_reps, u1_col_reps, signature)
        to2 = cls.from_U1(tu1, row_reps, col_reps)
        assert abs(to2.norm() - lg.norm(arr)) <= _tol * lg.norm(
            arr
        ), "norm is not conserved in O2_SymmetricTensor cast"
        return to2

    @classmethod
    def from_U1(cls, tu1, row_reps, col_reps):
        """
        Assume tu1 has O(2) symmetry and its irreps are sorted according to O(2) rules.

        No check is made on tensor norm at the end, so that U(1) tensor with only
        Sz >= 0 blocks may be used to initialize O(2) tensor. Allows to use partial U(1)
        tensor as intermediate step to construct O(2) without the additional cost of
        constructing Sz < 0 blocks and benefit from Sz = 0 block splitting.
        """

        # may remove these checks to be able to use U(1) with combined rows and columns
        # when tu1 is a temporary object
        # just need signature as input instead of tu1.signature
        assert tu1.n_row_reps == len(row_reps)
        assert all(
            (_numba_O2_rep_to_U1(r) == tu1.row_reps[i]).all()
            for i, r in enumerate(row_reps)
        )
        assert len(tu1.col_reps) == len(col_reps)
        assert all(
            (_numba_O2_rep_to_U1(r) == tu1.col_reps[i]).all()
            for i, r in enumerate(col_reps)
        )

        blocks = []
        block_irreps = []
        i0 = tu1.block_irreps.searchsorted(0)
        if tu1.block_irreps[i0] == 0:  # tu1 is O(2) => i0 < len(tu1.block_irreps)
            pro, pre = _get_b0_projectors(row_reps, tu1.get_row_representation())
            pco, pce = _get_b0_projectors(col_reps, tu1.get_column_representation())
            if pro.getnnz() and pco.getnnz():
                block_irreps.append(-1)
                b0o = custom_sparse_product(pro, tu1.blocks[i0], pco.T)
                blocks.append(b0o)
            if pre.getnnz() and pce.getnnz():
                block_irreps.append(0)
                b0e = custom_sparse_product(pre, tu1.blocks[i0], pce.T)
                blocks.append(b0e)
            assert abs(
                np.sqrt(sum(lg.norm(b) ** 2 for b in blocks)) - lg.norm(tu1.blocks[i0])
            ) <= _tol * lg.norm(tu1.blocks[i0]), "b0 splitting does not preserve norm"
            i0 += 1
        block_irreps.extend(tu1.block_irreps[i0:])
        blocks.extend(tu1.blocks[i0:])
        return cls(row_reps, col_reps, blocks, block_irreps, tu1.signature)

    def toarray(self, as_matrix=False):
        return self.toU1().toarray(as_matrix=as_matrix)

    def toabelian(self):
        return self.toU1()

    def _generate_neg_sz_blocks(self):
        # indexing is slightly faster with unsigned int
        # use uint64 everywhere to avoid integer mixing
        u1_row_reps = [None] * self._nrr
        rmaps = [None] * self._nrr
        rsigns = [None] * self._nrr
        for i, r in enumerate(self._row_reps):
            u1_row_reps[i] = _numba_O2_rep_to_U1(r)
            rmaps[i], rsigns[i] = _numba_get_reflection_perm_sign(r)

        shr = np.array(self.shape[: self._nrr], dtype=np.uint64)
        row_cp = np.array([np.uint64(1), *shr[-1:0:-1]]).cumprod()[::-1]
        u1_combined_row = U1_SymmetricTensor.combine_representations(
            u1_row_reps, self._signature[: self._nrr]
        )

        ncr = len(self._col_reps)
        u1_col_reps = [None] * ncr
        cmaps = [None] * ncr
        csigns = [None] * ncr
        for i, r in enumerate(self._col_reps):
            u1_col_reps[i] = _numba_O2_rep_to_U1(r)
            cmaps[i], csigns[i] = _numba_get_reflection_perm_sign(r)

        shc = np.array(self.shape[self._nrr :], dtype=np.uint64)
        col_cp = np.array([np.uint64(1), *shc[-1:0:-1]]).cumprod()[::-1]
        u1_combined_col = U1_SymmetricTensor.combine_representations(
            u1_col_reps, ~self._signature[self._nrr :]
        )

        blocks = []
        isz = self._nblocks - 1
        while isz > -1 and self._block_irreps[isz] > 0:
            sz = self._block_irreps[isz]
            # it is faster to map to Sz-reflected inside the loop.
            rsz_mat = (u1_combined_row == sz).nonzero()[0].view(np.uint64)  # Sz states
            rsz_t = (rsz_mat // row_cp[:, None]).T % shr  # multi-index form
            rszb_mat = np.zeros((rsz_mat.size,), dtype=np.uint64)
            rsign = np.ones((rsz_mat.size,), dtype=np.int8)
            for i, r in enumerate(self._row_reps):
                rszb_mat += rmaps[i][rsz_t[:, i]] * row_cp[i]  # map to spin reversed
                rsign *= rsigns[i][rsz_t[:, i]]

            csz_mat = (u1_combined_col == sz).nonzero()[0].view(np.uint64)
            csz_t = (csz_mat // col_cp[:, None]).T % shc  # multi-index form
            cszb_mat = np.zeros((csz_mat.size,), dtype=np.uint64)
            csign = np.ones((csz_mat.size,), dtype=np.int8)
            for i, r in enumerate(self._col_reps):
                cszb_mat += cmaps[i][csz_t[:, i]] * col_cp[i]  # map to spin reversed
                csign *= csigns[i][csz_t[:, i]]

            rso = rszb_mat.argsort().argsort().view(np.uint64)
            cso = cszb_mat.argsort().argsort().view(np.uint64)
            b = _numba_generate_block(rso, cso, self._blocks[isz], rsign, csign)
            blocks.append(b)
            isz -= 1

        return blocks

    def toO2(self):
        return self

    def toU1(self):
        # Sz < 0 blocks
        u1_row_reps = tuple(_numba_O2_rep_to_U1(r) for r in self._row_reps)
        u1_col_reps = tuple(_numba_O2_rep_to_U1(r) for r in self._col_reps)
        blocks = self._generate_neg_sz_blocks()
        block_sz = -self._block_irreps[::-1]

        # Sz = 0 blocks (may not exist)
        if self._block_irreps[0] < 1:
            sz_values = U1_SymmetricTensor.combine_representations(
                u1_row_reps, self._signature[: self._nrr]
            )
            pro, pre = _get_b0_projectors(self._row_reps, sz_values)
            sz_values = U1_SymmetricTensor.combine_representations(
                u1_col_reps, ~self._signature[self._nrr :]
            )
            pco, pce = _get_b0_projectors(self._col_reps, sz_values)
            if self._block_irreps[0] == 0:  # no 0o block
                i1 = 1
                b0 = custom_sparse_product(pre.T, self._blocks[0], pce)
                block_sz = np.hstack((block_sz, self._block_irreps[1:]))
            elif self.nblocks > 1 and self._block_irreps[1] > 0:  # no 0e block
                i1 = 1
                b0 = custom_sparse_product(pro.T, self._blocks[0], pco)
                block_sz[-1] = 0
                block_sz = np.hstack((block_sz, self._block_irreps[1:]))
            else:
                i1 = 2
                b0 = custom_double_sparse_product(
                    pro.T, self._blocks[0], pco, pre.T, self._blocks[1], pce
                )
                block_sz = np.hstack((block_sz[:-1], self._block_irreps[2:]))
            blocks.append(b0)
        else:
            i1 = 0
            block_sz = np.hstack((block_sz, self._block_irreps))

        # Sz > 0 blocks
        blocks.extend(self._blocks[i1:])

        tu1 = U1_SymmetricTensor(
            u1_row_reps, u1_col_reps, blocks, block_sz, self._signature
        )
        assert abs(tu1.norm() - self.norm()) <= _tol * self.norm()
        return tu1

    def permutate(self, row_axes, col_axes):
        assert sorted(row_axes + col_axes) == list(range(self._ndim))

        # return early for identity or matrix transpose
        if row_axes == tuple(range(self._nrr)) and col_axes == tuple(
            range(self._nrr, self._ndim)
        ):
            return self
        if row_axes == tuple(range(self._nrr, self._ndim)) and col_axes == tuple(
            range(self._nrr)
        ):
            return self.T

        # construct new row and column representations
        axes = row_axes + col_axes
        nrr = len(row_axes)
        signature = []
        reps = []

        u1_row_reps = [None] * self._nrr
        rmaps = [None] * self._nrr
        rsigns = [None] * self._nrr
        for i, r in enumerate(self._row_reps):
            u1_row_reps[i] = _numba_O2_rep_to_U1(r)
            rmaps[i], rsigns[i] = _numba_get_reflection_perm_sign(r)

        ncr = len(self._col_reps)
        u1_col_reps = [None] * ncr
        cmaps = [None] * ncr
        csigns = [None] * ncr
        for i, r in enumerate(self._col_reps):
            u1_col_reps[i] = _numba_O2_rep_to_U1(r)
            cmaps[i], csigns[i] = _numba_get_reflection_perm_sign(r)

        old_u1_reps = u1_row_reps + u1_col_reps
        u1_reps = []
        for ax in axes:
            signature.append(self._signature[ax])
            u1_reps.append(old_u1_reps[ax])
            if ax < self._nrr:
                reps.append(self._row_reps[ax])
            else:
                reps.append(self._col_reps[ax - self._nrr])
        signature = np.array(signature)

        # efficient O(2) product allows to precompute block_sz fast
        new_row_o2_rep = self.combine_representations(reps[:nrr], signature[:nrr])
        new_col_o2_rep = self.combine_representations(reps[nrr:], signature[nrr:])
        block_sz = np.intersect1d(
            new_row_o2_rep[1], new_col_o2_rep[1], assume_unique=True
        )
        if block_sz[0] == -1:
            if block_sz.size > 1 and block_sz[1] == 0:
                block_sz = block_sz[1:]
            else:
                block_sz[0] = 0

        old_row_sz = U1_SymmetricTensor.combine_representations(
            old_u1_reps[: self._nrr], self._signature[: self._nrr]
        )
        old_col_sz = U1_SymmetricTensor.combine_representations(
            old_u1_reps[self._nrr :], ~self._signature[self._nrr :]
        )
        new_row_sz = U1_SymmetricTensor.combine_representations(
            u1_reps[:nrr], signature[:nrr]
        )
        new_col_sz = U1_SymmetricTensor.combine_representations(
            u1_reps[nrr:], ~signature[nrr:]
        )

        # construct Sz = 0 block (may not exist)
        if self._block_irreps[0] < 1:
            pro, pre = _get_b0_projectors(self._row_reps, old_row_sz)
            pco, pce = _get_b0_projectors(self._col_reps, old_col_sz)
            if self._block_irreps[0] == 0:  # no 0o block
                b0 = custom_sparse_product(pre.T, self._blocks[0], pce)
                old_block_sz = self._block_irreps
                old_blocks = (b0, *self._blocks[1:])
            elif self.nblocks > 1 and self._block_irreps[1] > 0:  # no 0e block
                b0 = custom_sparse_product(pro.T, self._blocks[0], pco)
                old_block_sz = self._block_irreps.copy()
                old_block_sz[0] = 0
                old_blocks = (b0, *self._blocks[1:])
            else:
                b0 = custom_double_sparse_product(
                    pro.T, self._blocks[0], pco, pre.T, self._blocks[1], pce
                )
                old_block_sz = self._block_irreps[1:]
                old_blocks = (b0, *self._blocks[2:])
            del b0, pro, pre, pco, pce
        else:
            old_blocks = self._blocks
            old_block_sz = self._block_irreps

        old_blocks = tuple(np.ascontiguousarray(b) for b in old_blocks)  # numba C/F
        blocks = _numba_O2_transpose(
            np.array(self._shape, dtype=np.uint64),
            old_blocks,
            old_block_sz,
            old_row_sz,
            old_col_sz,
            axes,
            block_sz,
            new_row_sz,
            new_col_sz,
            tuple(rmaps),
            tuple(rsigns),
            tuple(cmaps),
            tuple(csigns),
        )
        del old_blocks, old_row_sz, old_col_sz, new_row_sz, new_col_sz
        tu1 = U1_SymmetricTensor(  # blocks Sz<0 are missing (will not be read)
            u1_reps[:nrr], u1_reps[nrr:], blocks, block_sz, signature
        )
        tp = self.from_U1(tu1, reps[:nrr], reps[nrr:])
        assert abs(tp.norm() - self.norm()) <= _tol * self.norm(), "norm is different"
        return tp

    @property
    def T(self):
        b_neg = tuple(b.T for b in reversed(self._generate_neg_sz_blocks()))
        b0 = tuple(b.T for b in self._blocks[: self._block_irreps.searchsorted(1)])
        blocks = b0 + b_neg
        s = self._signature[np.arange(-self._ndim + self._nrr, self._nrr) % self._ndim]
        return type(self)(self._col_reps, self._row_reps, blocks, self._block_irreps, s)

    def group_conjugated(self):
        signature = ~self._signature
        blocks = tuple(self._generate_neg_sz_blocks()[::-1])
        blocks = self._blocks[: self._block_irreps.searchsorted(1)] + blocks
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, signature
        )

    def update_signature(self, sign_update):
        # same as abelian case, bending an index to the left or to the right makes no
        # difference, signs can be ignored.
        up = np.asarray(sign_update, dtype=bool)
        assert up.shape == (self._ndim,)
        row_reps = list(self._row_reps)
        col_reps = list(self._col_reps)
        for i in up.nonzero()[0]:
            if i < self._nrr:
                row_reps[i] = self.conjugate_representation(row_reps[i])
            else:
                j = i - self._nrr
                col_reps[j] = self.conjugate_representation(col_reps[j])
        self._row_reps = tuple(row_reps)
        self._col_reps = tuple(col_reps)
        self._signature = self._signature ^ up
        assert self.check_blocks_fit_representations()

    def check_blocks_fit_representations(self):
        assert self._block_irreps.size == self._nblocks
        assert len(self._blocks) == self._nblocks
        row_rep = self.get_row_representation()
        col_rep = self.get_column_representation()
        r_indices = row_rep[1].searchsorted(self._block_irreps)
        c_indices = col_rep[1].searchsorted(self._block_irreps)
        assert (row_rep[1, r_indices] == self._block_irreps).all()
        assert (col_rep[1, c_indices] == self._block_irreps).all()
        for bi in range(self._nblocks):
            nr = row_rep[0, r_indices[bi]]
            nc = col_rep[0, c_indices[bi]]
            assert nr > 0
            assert nc > 0
            assert self._blocks[bi].shape == (nr, nc)
        return True

    def merge_legs(self, i1, i2):
        raise NotImplementedError("To do!")
