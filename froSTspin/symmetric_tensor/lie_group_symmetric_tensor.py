import numba
import numpy as np
import scipy.linalg as lg

from froSTspin.misc_tools.numba_tools import set_readonly_flag

from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor


@numba.njit
def _numba_transpose_reshape(old_mat, r1, r2, c1, c2, old_nrr, old_tensor_shape, perm):
    """
    numba version of
    old_mat[r1:r2, c1:c2].reshape(old_tensor_shape).transpose(perm)
    """
    NDIM = len(perm)

    old_tensor_strides = np.empty((NDIM,), dtype=np.int64)
    old_tensor_strides[old_nrr - 1] = old_mat.strides[0]
    old_tensor_strides[: old_nrr - 1] = (
        old_mat.strides[0] * np.cumprod(old_tensor_shape[old_nrr - 1 : 0 : -1])[::-1]
    )
    old_tensor_strides[old_nrr : NDIM - 1] = (
        old_mat.itemsize * np.cumprod(old_tensor_shape[NDIM - 1 : old_nrr : -1])[::-1]
    )
    old_tensor_strides[NDIM - 1] = old_mat.itemsize

    new_tensor_shape = np.empty((NDIM,), dtype=np.int64)
    new_tensor_strides = np.empty((NDIM,), dtype=np.int64)
    for i in range(NDIM):
        new_tensor_shape[i] = old_tensor_shape[perm[i]]
        new_tensor_strides[i] = old_tensor_strides[perm[i]]

    sht = numba.np.unsafe.ndarray.to_fixed_tuple(new_tensor_shape, NDIM)
    stridest = numba.np.unsafe.ndarray.to_fixed_tuple(new_tensor_strides, NDIM)
    permuted = np.lib.stride_tricks.as_strided(
        old_mat[r1:r2, c1:c2], shape=sht, strides=stridest
    )
    return permuted


@numba.njit(parallel=True)
def fill_blocks_out(
    old_matrix_blocks,
    simple_block_indices,
    idirb,
    idicb,
    idorb,
    idocb,
    edir,
    edic,
    edor,
    edoc,
    existing_matrix_block_inds,
    slices_ir,
    slices_ic,
    slices_or,
    slices_oc,
    old_row_external_mult,
    old_col_external_mult,
    isometry_in_blocks,
    data_perm,
):
    dtype = old_matrix_blocks[0].dtype
    n_new_matrix_blocks = slices_or.shape[1]
    NDIMP2 = len(data_perm)
    old_nrrp1 = data_perm[1]

    new_matrix_blocks = [
        np.zeros((slices_or[-1, i], slices_oc[-1, i]), dtype=dtype)
        for i in range(n_new_matrix_blocks)
    ]

    for i_simple_block in numba.prange(len(simple_block_indices)):
        # edor = external degeneracy out row
        # idicb = internal degeneracy in column block

        i_old_row, i_old_col, i_new_row, i_new_col = simple_block_indices[
            i_simple_block
        ]
        edir_ele = edir[i_old_row]
        edic_ele = edic[i_old_col]
        ele_sh = np.empty((NDIMP2,), dtype=np.int64)
        ele_sh[1:old_nrrp1] = old_row_external_mult[i_old_row]
        ele_sh[old_nrrp1 + 1 :] = old_col_external_mult[i_old_col]
        edor_ele = edor[i_new_row]
        edoc_ele = edoc[i_new_col]
        ext_mult = edor_ele * edoc_ele

        new_simple_block = np.zeros(
            ((idorb[i_new_row] * idocb[i_new_col]).sum(), ext_mult), dtype=dtype
        )

        for i_existing_sector, i_sector in enumerate(
            existing_matrix_block_inds
        ):  # filter missing blocks
            struct_mult_sector = idirb[i_old_row, i_sector] * idicb[i_old_col, i_sector]
            # need to check if this block_irrep_in appears in elementary_block
            if struct_mult_sector > 0:
                r2 = slices_ir[i_old_row, i_sector]
                c2 = slices_ic[i_old_col, i_sector]
                # there are two operations: changing basis with elementary AND
                # swapping axes in external degeneracy part. Transpose tensor BEFORE
                # applying unitary to do only one transpose

                ele_sh[0] = idirb[i_old_row, i_sector]
                ele_sh[old_nrrp1] = idicb[i_old_col, i_sector]

                # initial tensor shape = (
                # internal degeneracy in row block,
                # *external degeneracies per in row axes,
                # internal degeneracy in col block,
                # *external degeneracies per in col axes)
                #
                # transpose to shape = (
                # internal degeneracy in row block,
                # internal degeneracy in col, block,
                # *external degeneracies per OUT row axes,
                # *external degeneracies per OUT col axes)

                swapped_old_sym_block = _numba_transpose_reshape(
                    old_matrix_blocks[i_existing_sector],
                    r2 - edir_ele * idirb[i_old_row, i_sector],
                    r2,
                    c2 - edic_ele * idicb[i_old_col, i_sector],
                    c2,
                    old_nrrp1,
                    ele_sh,
                    data_perm,
                )

                swapped_old_sym_mat = swapped_old_sym_block.copy().reshape(
                    struct_mult_sector, ext_mult
                )

                # convention: iso_iblock is sliced irrep-wise on its columns = "IN"
                # but not for its rows = "OUT"
                # meaning it is applied to "IN" irrep block data, but generates data
                # for all OUT irrep blocks
                unitary_rows = isometry_in_blocks[i_simple_block, i_sector]
                new_simple_block += unitary_rows @ swapped_old_sym_mat

        # transpose new_simple_block only after every IN blocks have been processed
        oshift = 0
        for ibo in range(n_new_matrix_blocks):
            idorb_ele = idorb[i_new_row, ibo]
            idocb_ele = idocb[i_new_col, ibo]
            idob = idorb_ele * idocb_ele

            if idob > 0:
                new_symmetric_block = new_simple_block[oshift : oshift + idob]
                sh = (idorb_ele, idocb_ele, edor_ele, edoc_ele)
                new_symmetric_block = new_symmetric_block.reshape(sh).transpose(
                    0, 2, 1, 3
                )
                sh2 = (sh[0] * sh[2], sh[1] * sh[3])
                new_symmetric_block = new_symmetric_block.copy().reshape(sh2)

                # different IN block irreps may contribute: need +=
                sor2 = slices_or[i_new_row, ibo]
                sor1 = sor2 - sh2[0]
                soc2 = slices_oc[i_new_col, ibo]
                soc1 = soc2 - sh2[1]
                new_matrix_blocks[ibo][sor1:sor2, soc1:soc2] = new_symmetric_block
                oshift += idob

    return new_matrix_blocks


@numba.njit
def _numba_compute_external_degen(reps):
    """
    Compute external degeneracies on each axis for all combination of elementary blocks

    Parameters
    ----------
    reps : tuple of nr C-contiguous 2D int array
        Representations to fuse. Only the first row, corresponding to degeneracies, is
        read.

    Returns
    -------
    external_degen : (n_ele, nr) int array
        External degeneracies on each axis for each elementary block.
    """
    nr = len(reps)
    shapes = np.array([r.shape[1] for r in reps])
    strides = np.empty((nr,), dtype=np.int64)
    strides[-1] = 1
    strides[:-1] = shapes[:0:-1].cumprod()[::-1]
    n_ele_blocks = strides[0] * shapes[0]
    external_degen = np.empty((n_ele_blocks, nr), dtype=np.int64)
    for i in np.arange(n_ele_blocks):
        for j in range(nr):
            k = i // strides[j] % shapes[j]
            external_degen[i, j] = reps[j][0, k]
    return external_degen


class LieGroupSymmetricTensor(NonAbelianSymmetricTensor):
    r"""
    Efficient storage and manipulation for a tensor with non-abelian symmetry defined
    by a Lie group. Axis permutation is done using isometries defined by fusion trees of
    representations.

    Impose a tree structure splitted between rows and columns. Each tree is maximally
    unbalanced, with only one leg with depth.

                    singlet space
                    /          \
                   /            \
                rprod          cprod
                 /               /
                /\              /\
               /\ \            /\ \
             row_reps        col_reps
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    # every method is group-specific

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    _structural_data_dic = NotImplemented  # permute information for a given tensor
    _unitary_dic = NotImplemented  # unitaries for a given elementary block

    @classmethod
    def load_isometries(cls, savefile):
        raise NotImplementedError("TODO!")
        """
        with np.load(savefile) as fin:
            if fin["_ST_symmetry"] != cls.symmetry:
                raise ValueError("Savefile symmetry does not match SymmetricTensor")
            for i in range(fin["_ST_n_iso"]):
                key = tuple(fin[f"_ST_iso_{i}_key"])
                block_irreps = fin[f"_ST_iso_{i}_block_irreps"]
                blocks = []
                for j in range(block_irreps.size):
                    data = fin[f"_ST_iso_{i}_{j}_data"]
                    indices = fin[f"_ST_iso_{i}_{j}_indices"]
                    indptr = fin[f"_ST_iso_{i}_{j}_indptr"]
                    shape = fin[f"_ST_iso_{i}_{j}_shape"]
                    b = ssp.csc_array((data, indices, indptr), shape=shape)
                    blocks.append(b)
                cls._isometry_dic[key] = (np.array(block_irreps), blocks)
        """

    @classmethod
    def save_isometries(cls, savefile):
        raise NotImplementedError("TODO!")
        """
        data = {"_ST_symmetry": cls.symmetry, "_ST_n_iso": len(cls._isometry_dic)}
        # keys may be very long, may get into trouble as valid archive name beyond 250
        # char. Just count values and save keys as arrays.
        for i, (k, v) in enumerate(cls._isometry_dic.items()):
            data[f"_ST_iso_{i}_key"] = np.array(k, dtype=int)
            data[f"_ST_iso_{i}_block_irreps"] = v[0]
            assert len(v[1]) == v[0].size
            for j, b in enumerate(v[1]):
                data[f"_ST_iso_{i}_{j}_data"] = b.data
                data[f"_ST_iso_{i}_{j}_indices"] = b.indices
                data[f"_ST_iso_{i}_{j}_indptr"] = b.indptr
                data[f"_ST_iso_{i}_{j}_shape"] = b.shape
        np.savez_compressed(savefile, **data)
        """

    ####################################################################################
    # Lie group shared symmetry implementation
    ####################################################################################
    @classmethod
    def compute_clebsch_gordan_tree(cls, rep_in, signature, *, target_irreps=None):
        r"""
        Construct chained Clebsch-Gordan fusion tensor for representations *rep_in with
        signatures signatures. Truncate final representation at max_irrep.

        Tree structure: only first leg has depth
                    product
                      /
                    ...
                    /
                   /\
                  /  \
                 /\   \
                /  \   \
               1    2   3 ...

        Parameters
        ----------
        rep_in : enum of n 2D int array
            SU(2) representations to fuse.
        signature : (n,) bool array
            Representation signatures.
        target_irreps : int array
            Limit number of computed CG trees by targetting some irreps only.

        Returns
        -------
        ret : 2D float array
            CG projector fusing rep_in on sum of irreps, truncated up to max_irrep.
            Reshaped as a 2D matrix (input_dimension, output_dimension).
        """
        raise NotImplementedError("Must be defined in derived class")

    @classmethod
    def sliced_elementary_trees(cls, reps, signature, *, target_irreps=None):
        r"""
        Construct Clebsch-Gordon trees for all elementary blocks of a list of reducible
        representations, for all irreps in target.

           irrep0         irrep1         irrep2
            /              /              /
           /\             /\             /\
          /\ \           /\ \           /\ \
         /\ \ \         /\ \ \         /\ \ \
          reps           reps           reps

        Parameters
        ----------
        reps : enumerable of n representations
            Representations to decompose in elementary blocks and fuse.
        signature : (n,) bool array
            Signature on each axis.
        target_irreps : 1D integer array or None
            Irreps to consider in total representation. If None, all irreps are
            conserved.
        """
        # assume irrep is just one integer
        # do not assume irrep = dimension(irrep)
        if target_irreps is None:  # keep all irreps
            total_rep = cls.combine_representations(reps, signature)
            target_irreps = total_rep[1]

        elementary_block_per_axis = np.array([r.shape[1] for r in reps])
        n_ele_blocks = elementary_block_per_axis.prod()
        n_irreps = len(target_irreps)
        internal_degeneracies = np.zeros((n_ele_blocks, n_irreps), dtype=int)
        ele_trees = np.empty((n_ele_blocks, n_irreps), dtype=object)

        for i_ele in range(n_ele_blocks):
            # construct CG tree for this elementary block
            mul_ind = np.unravel_index(i_ele, elementary_block_per_axis)
            ele_reps = tuple(
                np.array([[1], [r[1, mul_ind[i]]]]) for i, r in enumerate(reps)
            )
            # some trees will actually be empty, truncated to nothing by max_irrep
            # we cannot know it here as we are still symmetry-agnostic
            # cls.compute_clebsch_gordan_tree must detect it and return rep = [[], []]
            rep, tree = cls.compute_clebsch_gordan_tree(
                ele_reps, signature, target_irreps=target_irreps
            )

            # slice according to irrep sectors
            k = 0
            sh = [cls.irrep_dimension(r[1, 0]) for r in ele_reps] + [-1, -1]
            for i in range(rep.shape[1]):
                internal_degen = rep[0, i]
                irrep = rep[1, i]
                irrep_dim = cls.irrep_dimension(irrep)
                sector_dim = internal_degen * irrep_dim
                bi = target_irreps.searchsorted(irrep)
                if bi < n_irreps and irrep == target_irreps[bi]:
                    sh[-2] = internal_degen
                    sh[-1] = irrep_dim
                    proj = np.ascontiguousarray(tree[:, k : k + sector_dim].reshape(sh))
                    ele_trees[i_ele, bi] = proj
                    internal_degeneracies[i_ele, bi] = internal_degen
                k += sector_dim
            assert k == tree.shape[1]
        return internal_degeneracies, ele_trees

    @classmethod
    def _compute_structural_data(cls, in_reps, nrr_in, axes, nrr_out, signature_in):
        """
        Parameters
        ----------
        in_reps : tuple of representations
            Input representations
        nrr_in : int
            Number of input row representations.
        axes : 1D integer array
            Permutation as a 1D axis
        nrr_out : int
            Number of axes considered as row
        signature_in : bool array
            Input signature.

        Returns
        -------
        structural data
        """
        # Purely structural. Can be precomputed.
        # use key as input?

        # check input
        ndim = len(in_reps)
        assert axes.shape == (ndim,)
        assert 0 < nrr_in < ndim
        assert 0 < nrr_out < ndim
        assert signature_in.shape == (ndim,)

        # find block irreps in and out
        block_irreps_in, _ = cls.get_block_sizes(
            in_reps[:nrr_in], in_reps[nrr_in:], signature_in
        )
        nblocks_in = len(block_irreps_in)
        if nblocks_in == 0:
            raise ValueError("Representations do not allow any block")

        signature_out = signature_in[axes]
        out_row_reps = tuple(in_reps[i] for i in axes[:nrr_out])
        out_col_reps = tuple(in_reps[i] for i in axes[nrr_out:])
        block_irreps_out, _ = cls.get_block_sizes(
            out_row_reps, out_col_reps, signature_out
        )
        n_new_matrix_blocks = len(block_irreps_out)

        elementary_block_per_axis = np.array([r.shape[1] for r in in_reps])
        n_ele_ri = elementary_block_per_axis[:nrr_in].prod()
        n_ele_ci = elementary_block_per_axis[nrr_in:].prod()
        n_ele_ro = elementary_block_per_axis[axes[:nrr_out]].prod()
        n_ele_co = elementary_block_per_axis[axes[nrr_out:]].prod()

        idirb = np.zeros((n_ele_ri, nblocks_in), dtype=int)
        idicb = np.zeros((n_ele_ci, nblocks_in), dtype=int)
        idorb = np.zeros((n_ele_ro, n_new_matrix_blocks), dtype=int)
        idocb = np.zeros((n_ele_co, n_new_matrix_blocks), dtype=int)
        cg_trees_ri = [None] * n_ele_ri
        cg_trees_ci = [None] * n_ele_ci
        cg_trees_ro = [None] * n_ele_ro
        cg_trees_co = [None] * n_ele_co
        simple_block_indices = []

        key0 = np.empty((2 * ndim + 3,), dtype=np.int64)
        key0[ndim] = nrr_in
        key0[ndim + 1] = nrr_out
        key0[ndim + 2] = (1 << np.arange(ndim)) @ signature_in
        key0[ndim + 3 :] = axes
        singlet = cls.singlet()[1, 0]

        # convention: return elementary unitary blocked sliced for IN irrep blocks
        non_writable_array = np.empty((2, 2))
        set_readonly_flag(non_writable_array)
        isometry_blocks = numba.typed.Dict.empty(
            key_type=numba.types.UniTuple(numba.types.int64, 2),
            value_type=numba.typeof(non_writable_array),
        )

        # need to add idirb and idicb axes in permutation
        perm = np.empty((ndim + 2,), dtype=int)
        perm[:ndim] = (axes >= nrr_in) + axes
        perm[ndim] = nrr_in
        perm[ndim + 1] = ndim + 1

        # strides to recover ir/ic_in/out from i_ele
        strides = np.zeros((4, ndim), dtype=int)
        strides[0, nrr_in - 1] = 1
        for i in reversed(range(1, nrr_in)):
            strides[0, i - 1] = elementary_block_per_axis[i] * strides[0, i]
        strides[1, ndim - 1] = 1
        for i in reversed(range(nrr_in + 1, ndim)):
            strides[1, i - 1] = elementary_block_per_axis[i] * strides[1, i]
        strides[2, axes[nrr_out - 1]] = 1
        for i in reversed(range(1, nrr_out)):
            strides[2, axes[i - 1]] = (
                elementary_block_per_axis[axes[i]] * strides[2, axes[i]]
            )
        strides[3, axes[ndim - 1]] = 1
        for i in reversed(range(nrr_out + 1, ndim)):
            strides[3, axes[i - 1]] = (
                elementary_block_per_axis[axes[i]] * strides[3, axes[i]]
            )

        sig_ri = signature_in[:nrr_in]
        sig_ci = ~signature_in[nrr_in:]
        sig_ro = signature_out[:nrr_out]
        sig_co = ~signature_out[nrr_out:]

        i_ele = 0
        for i0 in range(n_ele_ri * n_ele_ci):
            i_mul = np.array(np.unravel_index(i0, elementary_block_per_axis))
            reps_ei = [None] * ndim
            for i in range(ndim):
                irr = in_reps[i][1, i_mul[i]]
                reps_ei[i] = np.array([[1], [irr]])
                key0[i] = irr
            ri_rep = cls.combine_representations(reps_ei[:nrr_in], sig_ri)
            ci_rep = cls.combine_representations(reps_ei[nrr_in:], sig_ci)
            shared_ri = (ri_rep[1, :, None] == ci_rep[1]).nonzero()[0]

            if shared_ri.size:  # if the block contributes
                ele_inds = strides @ i_mul
                ir_in, ic_in, ir_out, ic_out = ele_inds
                simple_block_indices.append(ele_inds)
                irreps_in_ele = ri_rep[1, shared_ri]
                ele_in_binds = (irreps_in_ele[:, None] == block_irreps_in).nonzero()[1]
                reps_ero = tuple(reps_ei[i] for i in axes[:nrr_out])
                reps_eco = tuple(reps_ei[i] for i in axes[nrr_out:])

                if not idirb[ir_in].any():
                    inds_e, inds_f = (ri_rep[1, :, None] == block_irreps_in).nonzero()
                    idirb[ir_in, inds_f] = ri_rep[0, inds_e]
                if not idicb[ic_in].any():
                    inds_e, inds_f = (ci_rep[1, :, None] == block_irreps_in).nonzero()
                    idicb[ic_in, inds_f] = ci_rep[0, inds_e]
                if not idorb[ir_out].any():
                    ro_rep = cls.combine_representations(reps_ero, sig_ro)
                    inds_e, inds_f = (ro_rep[1, :, None] == block_irreps_out).nonzero()
                    idorb[ir_out, inds_f] = ro_rep[0, inds_e]
                if not idocb[ic_out].any():
                    co_rep = cls.combine_representations(reps_eco, sig_co)
                    inds_e, inds_f = (co_rep[1, :, None] == block_irreps_out).nonzero()
                    idocb[ic_out, inds_f] = co_rep[0, inds_e]

                # prune singlets
                kept = (key0[:ndim] != singlet).nonzero()[0]
                if kept.size == ndim:
                    key = tuple(key0)
                else:
                    nrr_kept = (kept < nrr_in).sum()
                    sig_kept = (1 << np.arange(kept.size)) @ signature_in[kept]
                    kept_axes = axes.argsort()[kept]
                    nrr_out_kept = (kept_axes < nrr_out).sum()
                    kept_perm = kept_axes.argsort()

                    # get a new key corresponding to same reps and perm without singlet
                    key = (
                        *key0[kept],
                        nrr_kept,
                        nrr_out_kept,
                        sig_kept,
                        *kept_perm,
                    )

                try:  # maybe we already computed the unitary
                    ele_unitary = cls._unitary_dic[key]
                except KeyError:
                    # we need to use block_irreps_in as target irreps, not ele_irreps_in
                    # as some block may be absent in ic_in, but not for ic_in + 1
                    if cg_trees_ri[ir_in] is None:
                        idirb_e, trees = cls.sliced_elementary_trees(
                            reps_ei[:nrr_in],
                            sig_ri,
                            target_irreps=block_irreps_in,
                        )
                        cg_trees_ri[ir_in] = trees[0]  # only 1 ele block
                        assert (idirb_e == idirb[ir_in]).all()
                    if cg_trees_ci[ic_in] is None:
                        idicb_e, trees = cls.sliced_elementary_trees(
                            reps_ei[nrr_in:],
                            sig_ci,
                            target_irreps=block_irreps_in,
                        )
                        cg_trees_ci[ic_in] = trees[0]
                        assert (idicb_e == idicb[ic_in]).all()
                    if cg_trees_ro[ir_out] is None:
                        idorb_e, trees = cls.sliced_elementary_trees(
                            reps_ero,
                            sig_ro,
                            target_irreps=block_irreps_out,
                        )
                        cg_trees_ro[ir_out] = trees[0]
                        assert (idorb_e == idorb[ir_out]).all()
                    if cg_trees_co[ic_out] is None:
                        idocb_e, trees = cls.sliced_elementary_trees(
                            reps_eco,
                            sig_co,
                            target_irreps=block_irreps_out,
                        )
                        assert (idocb_e == idocb[ic_out]).all()
                        cg_trees_co[ic_out] = trees[0]

                    ele_unitary = cls.overlap_cg_trees(
                        cg_trees_ri[ir_in],
                        cg_trees_ci[ic_in],
                        cg_trees_ro[ir_out],
                        cg_trees_co[ic_out],
                        perm,
                    )
                    # save elementary block unitary for later
                    cls._unitary_dic[key] = ele_unitary

                assert idirb[ir_in] @ idicb[ic_in] == idorb[ir_out] @ idocb[ic_out]
                # map ele in_blocks_index to full tensor in_blocks_index
                for i, eub in enumerate(ele_unitary):
                    bi_in = ele_in_binds[i]
                    isometry_blocks[i_ele, bi_in] = eub
                i_ele += 1

        simple_block_indices = np.array(simple_block_indices)
        assert (
            (idirb[simple_block_indices[:, 0]] * idicb[simple_block_indices[:, 1]]).sum(
                axis=1
            )
            == (
                idorb[simple_block_indices[:, 2]] * idocb[simple_block_indices[:, 3]]
            ).sum(axis=1)
        ).all(), "number of coefficient not conserved"

        set_readonly_flag(
            simple_block_indices, idirb, idicb, idorb, idocb, block_irreps_out
        )

        structual_data = (
            simple_block_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            block_irreps_out,
            isometry_blocks,
        )
        return structual_data

    @classmethod
    def overlap_cg_trees(
        cls,
        cg_ri,
        cg_ci,
        cg_ro,
        cg_co,
        perm,
    ):
        # cg_ri and cg_ci are list of trees sliced by irrep. The length for out may be
        # larger than the number of allowed block irreps.
        # if a given irrep appears only in at least one of the two trees ouptut, the
        # tree will be replaced by None.
        # ri trees have shape (*dim_ele_r, idirb, irr)
        assert len(cg_ri) == len(cg_ci)
        n_blocks_in = len(cg_ri)
        ele_unitary_blocks = []

        # precompute out block_proj
        assert len(cg_ro) == len(cg_co)
        out_proj_block = []
        dims_out = []
        di = 0
        for rtree, ctree in zip(cg_ro, cg_co, strict=True):
            if rtree is not None and ctree is not None:
                assert rtree.shape[-1] == ctree.shape[-1]
                dim = rtree.shape[-1]  # dim(irrep)
                dims_out.append(dim)
                rt = rtree.reshape(-1, dim)  # shape (dim_ele_r * idorb, irrep_dim)
                ct = ctree.reshape(-1, dim)  # shape (irrep_dim, dim_ele_c * idocb)
                proj_ob = rt @ ct.T  # shape (dim_ele_r * idorb, dim_ele_c * idocb)
                sh = (
                    rt.shape[0] // rtree.shape[-2],
                    rtree.shape[-2],
                    ct.shape[0] // ctree.shape[-2],
                    ctree.shape[-2],
                )
                proj_ob = proj_ob.reshape(sh).swapaxes(1, 2)
                nsh = (sh[0] * sh[2], sh[1] * sh[3])  # shape (dim_ele, idorb * idocb)
                out_proj_block.append(proj_ob.reshape(nsh).T)
                di += nsh[1]

        n_blocks_out = len(dims_out)
        for bi_in in range(n_blocks_in):
            rtree = cg_ri[bi_in]  # shape (*dim_ele_r, idirb, irr)
            ctree = cg_ci[bi_in]  # shape (*dim_ele_c, idicb, irr)
            if rtree is not None and ctree is not None:
                assert rtree.shape[-1] == ctree.shape[-1]
                irrep_dim = rtree.shape[-1]
                rt = rtree.reshape(-1, irrep_dim)
                ct = ctree.reshape(-1, irrep_dim)
                proj_ib = rt @ ct.T  # shape (dim_ele_r * idirb, dim_ele_c * idicb)
                sh = rtree.shape[:-1] + ctree.shape[:-1]
                proj_ib = proj_ib.reshape(sh).transpose(perm)
                proj_ib = proj_ib.reshape(-1, rtree.shape[-2] * ctree.shape[-2])

                sh = (di, proj_ib.shape[1])
                isometry_bi = np.empty(sh)

                k = 0
                for bi_out in range(n_blocks_out):
                    isometry = out_proj_block[bi_out] @ proj_ib
                    # not a unitary: not square + sqrt(irrep_dim) not included
                    # with sqrt(irrep) and packed with matrices from other irrep
                    # blocks it becomes unitary => coefficients are of order 1,
                    # it is safe to delete numerical zeros according to absolute
                    # tolerance
                    isometry[np.abs(isometry) < 1e-14] = 0
                    dbo = isometry.shape[0]
                    isometry_bi[k : k + isometry.shape[0]] = isometry / dims_out[bi_out]
                    k += dbo

                assert k == isometry_bi.shape[0]
                set_readonly_flag(isometry_bi)
                ele_unitary_blocks.append(isometry_bi)

        return ele_unitary_blocks

    def _lie_permute_data(self, axes, new_nrr, structural_data):
        """
        Move data and construct new data blocks
        """
        # 3 nested loops: elementary blocks, rows, columns
        # change loop order?

        old_reps = self._row_reps + self._col_reps
        new_row_reps = tuple(old_reps[i] for i in axes[:new_nrr])
        new_col_reps = tuple(old_reps[i] for i in axes[new_nrr:])

        (
            simple_block_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            block_irreps_out,
            isometry_in_blocks,
        ) = structural_data

        old_row_external_mult = _numba_compute_external_degen(self._row_reps)
        old_col_external_mult = _numba_compute_external_degen(self._col_reps)

        edir = old_row_external_mult.prod(axis=1)
        edic = old_col_external_mult.prod(axis=1)
        edor = _numba_compute_external_degen(new_row_reps).prod(axis=1)
        edoc = _numba_compute_external_degen(new_col_reps).prod(axis=1)

        assert idorb.shape[1] == len(block_irreps_out)
        assert idocb.shape[1] == len(block_irreps_out)
        assert old_row_external_mult.shape[0] == idirb.shape[0]
        assert old_col_external_mult.shape[0] == idicb.shape[0]

        assert self._nblocks <= idirb.shape[1]
        if self._nblocks < idirb.shape[1]:  # if a block is missing
            # filter out missing blocks
            block_irreps_in, _ = self.get_block_sizes(
                self._row_reps, self._col_reps, self._signature
            )
            _, existing_matrix_block_inds = (
                self._block_irreps[:, None] == block_irreps_in
            ).nonzero()
            existing_matrix_block_inds = np.ascontiguousarray(
                existing_matrix_block_inds
            )
            # possible also filter simple_block_indices with (idirb.T @ idicb).nonzero()
            # probably not worth it
        else:
            existing_matrix_block_inds = np.arange(self._nblocks)

        slices_ir = (edir[:, None] * idirb).cumsum(axis=0)
        slices_ic = (edic[:, None] * idicb).cumsum(axis=0)
        slices_or = (edor[:, None] * idorb).cumsum(axis=0)
        slices_oc = (edoc[:, None] * idocb).cumsum(axis=0)

        # need to initalize blocks_out in case of missing blocks_in
        old_matrix_blocks = tuple(np.ascontiguousarray(b) for b in self._blocks)
        set_readonly_flag(
            edir,
            edic,
            edor,
            edoc,
            existing_matrix_block_inds,
            slices_ir,
            slices_ic,
            slices_or,
            slices_oc,
            old_row_external_mult,
            old_col_external_mult,
            *old_matrix_blocks,
        )

        data_perm = (0, self._nrr + 1, *(axes + 1 + (axes >= self._nrr)))
        new_matrix_blocks = fill_blocks_out(
            old_matrix_blocks,
            simple_block_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            edir,
            edic,
            edor,
            edoc,
            existing_matrix_block_inds,
            slices_ir,
            slices_ic,
            slices_or,
            slices_oc,
            old_row_external_mult,
            old_col_external_mult,
            isometry_in_blocks,
            data_perm,
        )
        return block_irreps_out, new_matrix_blocks

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################

    # private helper function
    @classmethod
    def _get_shifts_positions(cls, reps):
        """
        Define shifts to localize elementary block position in dense format
        """
        shifts = []
        elementary_position = np.empty((len(reps),), dtype=int)
        for i, rep in enumerate(reps):
            n = rep.shape[1]
            elementary_position[i] = n
            s = [0, *(cls.representation_dimension(rep[:, : j + 1]) for j in range(n))]
            shifts.append(s)
        return shifts, elementary_position

    @classmethod
    def _blocks_from_dense(cls, arr, row_reps, col_reps, signature):
        block_irreps, block_shapes_in = cls.get_block_sizes(
            row_reps, col_reps, signature
        )
        nrr = len(row_reps)
        ndim = nrr + len(col_reps)

        # define shifts to localize elementary block position in dense format
        rshifts, elementary_block_rows = cls._get_shifts_positions(row_reps)
        cshifts, elementary_block_cols = cls._get_shifts_positions(col_reps)

        # define permutation to bring together external degen and irrep dimension
        degen_irrep_perm = tuple(range(1, 2 * ndim, 2)) + tuple(range(0, 2 * ndim, 2))

        # precompute CG trees for row and columns. Filter for missing blocks.
        idirb, irb_trees = cls.sliced_elementary_trees(
            row_reps, signature[:nrr], target_irreps=block_irreps
        )
        idicb, icb_trees = cls.sliced_elementary_trees(
            col_reps, ~signature[nrr:], target_irreps=block_irreps
        )

        # precompute external degeneracies and irrep block shifts
        old_row_external_mult = _numba_compute_external_degen(row_reps)
        old_col_external_mult = _numba_compute_external_degen(col_reps)
        edir = old_row_external_mult.prod(axis=1)
        edic = old_col_external_mult.prod(axis=1)
        block_shifts_row = (edir[:, None] * idirb).cumsum(axis=0)
        block_shifts_col = (edic[:, None] * idicb).cumsum(axis=0)

        # initialize blocks sizes
        blocks = tuple(np.empty(sh, dtype=arr.dtype) for sh in block_shapes_in)
        irrep_dims = [cls.irrep_dimension(irr) for irr in block_irreps]

        # loop over row elementary blocks
        for i_old_row in range(idirb.shape[0]):
            ele_degen_dimensions = np.empty((2, ndim), dtype=int)
            ele_degen_dimensions[0, :nrr] = old_row_external_mult[i_old_row]
            irmul = np.unravel_index(i_old_row, elementary_block_rows)
            rslices = []
            for i in range(nrr):
                ele_degen_dimensions[1, i] = cls.irrep_dimension(
                    row_reps[i][1, irmul[i]]
                )
                rslices.append(slice(rshifts[i][irmul[i]], rshifts[i][irmul[i] + 1]))
            rdim = ele_degen_dimensions[1, :nrr].prod()

            # loop over column elementary blocks
            for i_old_col in range(idicb.shape[0]):
                idi = idirb[i_old_row].T @ idicb[i_old_col]
                if idi > 0:
                    ele_degen_dimensions[0, nrr:] = old_col_external_mult[i_old_col]
                    icmul = np.unravel_index(i_old_col, elementary_block_cols)
                    cslices = []
                    for i in range(ndim - nrr):
                        ele_degen_dimensions[1, nrr + i] = cls.irrep_dimension(
                            col_reps[i][1, icmul[i]]
                        )
                        cslices.append(
                            slice(cshifts[i][icmul[i]], cshifts[i][icmul[i] + 1])
                        )
                    edi = edir[i_old_row] * edic[i_old_col]
                    cdim = ele_degen_dimensions[1, nrr:].prod()

                    # split degen and structural
                    slices = tuple(rslices + cslices)
                    sh = ele_degen_dimensions.T.ravel()
                    data = arr[slices].reshape(sh)
                    assert rdim * cdim == ele_degen_dimensions[1].prod()
                    data = data.transpose(degen_irrep_perm).reshape(rdim * cdim, edi)
                    idib = idirb[i_old_row] * idicb[i_old_col]
                    for bi in idib.nonzero()[0]:
                        # construct CG projector on block irrep elementary sector
                        # see _tomatrix for a discussion of the contraction scheme

                        rtree = irb_trees[i_old_row, bi]
                        rtree = rtree.reshape(-1, rtree.shape[-1])
                        ctree = icb_trees[i_old_col, bi].reshape(-1, rtree.shape[1])
                        block_ele_proj = rtree @ ctree.T
                        sh = (rdim, idirb[i_old_row, bi], cdim, idicb[i_old_col, bi])
                        block_ele_proj = block_ele_proj.reshape(sh).swapaxes(1, 2)
                        block_ele_proj = block_ele_proj.reshape(rdim * cdim, idib[bi])

                        data_block = block_ele_proj.T @ data

                        # data_block still has a non-trivial structure due to inner
                        # degeneracies. Its shape is
                        # (int_degen_row, int_degen_col, ext_degen_row, ext_degen_col)
                        # we need to permute axes to reshape as
                        # (ext_degen_row * int_degen_row, ext_degen_col * int_degen_col)
                        sh = (
                            idirb[i_old_row, bi],
                            idicb[i_old_col, bi],
                            edir[i_old_row],
                            edic[i_old_col],
                        )
                        sh2 = (sh[0] * sh[2], sh[1] * sh[3])
                        data_block = data_block.reshape(sh).swapaxes(1, 2)
                        data_block = (data_block / irrep_dims[bi]).reshape(sh2)

                        rs = slice(
                            block_shifts_row[i_old_row, bi] - sh2[0],
                            block_shifts_row[i_old_row, bi],
                        )
                        cs = slice(
                            block_shifts_col[i_old_col, bi] - sh2[1],
                            block_shifts_col[i_old_col, bi],
                        )
                        blocks[bi][rs, cs] = data_block

        return blocks, block_irreps

    def _tomatrix(self):
        # define shifts to localize elementary block position in dense format
        rshifts, elementary_block_rows = self._get_shifts_positions(self._row_reps)
        cshifts, elementary_block_cols = self._get_shifts_positions(self._col_reps)

        # define permutation to bring together external degen and irrep dimension
        degen_irrep_perm = tuple(range(1, 2 * self._ndim, 2)) + tuple(
            range(0, 2 * self._ndim, 2)
        )
        reverse_perm = np.argsort(degen_irrep_perm)

        # precompute CG trees for row and columns. Filter for missing blocks.
        idorb, orb_trees = self.sliced_elementary_trees(
            self._row_reps,
            self._signature[: self._nrr],
            target_irreps=self._block_irreps,
        )
        idocb, ocb_trees = self.sliced_elementary_trees(
            self._col_reps,
            ~self._signature[self._nrr :],
            target_irreps=self._block_irreps,
        )

        # precompute external degeneracies and irrep block shifts
        external_degen_or = _numba_compute_external_degen(self._row_reps)
        external_degen_oc = _numba_compute_external_degen(self._col_reps)
        edor = external_degen_or.prod(axis=1)
        edoc = external_degen_oc.prod(axis=1)
        block_shifts_row = (edor[:, None] * idorb).cumsum(axis=0)
        block_shifts_col = (edoc[:, None] * idocb).cumsum(axis=0)

        # initialize dense array
        dtype = self.dtype
        arr = np.zeros(self._shape, dtype=dtype)

        # loop over row elementary blocks
        for i_new_row in range(idorb.shape[0]):
            ele_degen_dimensions = np.empty((2, self._ndim), dtype=int)
            ele_degen_dimensions[0, : self._nrr] = external_degen_or[i_new_row]
            irmul = np.unravel_index(i_new_row, elementary_block_rows)
            rslices = []
            for i in range(self._nrr):
                ele_degen_dimensions[1, i] = self.irrep_dimension(
                    self._row_reps[i][1, irmul[i]]
                )
                rslices.append(slice(rshifts[i][irmul[i]], rshifts[i][irmul[i] + 1]))
            rdim = ele_degen_dimensions[1, : self._nrr].prod()

            # loop over column elementary blocks
            for i_new_col in range(idocb.shape[0]):
                ido = idorb[i_new_row].T @ idocb[i_new_col]
                if ido > 0:  # if this elementary block is allowed
                    ele_degen_dimensions[0, self._nrr :] = external_degen_oc[i_new_col]
                    icmul = np.unravel_index(i_new_col, elementary_block_cols)
                    cslices = []
                    for i in range(self._ndim - self._nrr):
                        ele_degen_dimensions[1, self._nrr + i] = self.irrep_dimension(
                            self._col_reps[i][1, icmul[i]]
                        )
                        cslices.append(
                            slice(cshifts[i][icmul[i]], cshifts[i][icmul[i] + 1])
                        )
                    edo = edor[i_new_row] * edoc[i_new_col]
                    cdim = ele_degen_dimensions[1, self._nrr :].prod()
                    ele_dense = np.zeros((rdim * cdim, edo), dtype=dtype)
                    idob = idorb[i_new_row] * idocb[i_new_col]
                    for bi in idob.nonzero()[0]:
                        rs = slice(
                            block_shifts_row[i_new_row, bi]
                            - edor[i_new_row] * idorb[i_new_row, bi],
                            block_shifts_row[i_new_row, bi],
                            1,
                        )
                        cs = slice(
                            block_shifts_col[i_new_col, bi]
                            - edoc[i_new_col] * idocb[i_new_col, bi],
                            block_shifts_col[i_new_col, bi],
                            1,
                        )
                        sh = (
                            idorb[i_new_row, bi],
                            edor[i_new_row],
                            idocb[i_new_col, bi],
                            edoc[i_new_col],
                        )
                        data_block = self._blocks[bi][rs, cs].reshape(sh)
                        data_block = data_block.swapaxes(1, 2).reshape(idob[bi], edo)

                        # construct CG projector on block irrep elementary sector
                        # we start from 2 CG trees, one for the rows, one for the
                        # columns + data_block. dimb is block irrep dimenion.
                        #
                        #               edor             edoc
                        #                 \              /
                        #                  data_ele_block
                        #                 /              \
                        #               idorb           idocb
                        #
                        #
                        #     dimb  idorb              dimb   idocb
                        #        \  /                     \  /
                        #         \/                       \/
                        #         /                        /
                        #       ...                      ...
                        #       /                        /
                        #      /\                       /
                        #     /  \                     /
                        #    /\   \                   /\
                        #   /  \   \                 /  \
                        #  i1   i2  i3 ...          i4   i5  ...
                        #
                        #
                        # 2 ways of contracting this:
                        # data_block * (rtree * ctree), with complexity
                        #    prod_{r,c}(dim i_k) * dimb * ido
                        #  + edo * ido * prod_{r,c}(dim irr)
                        # or (data_block * rtree) * ctree with complexity
                        #    prod_{r}(dim i_k) * dimb * edo * ido
                        #  + edo * idocb * dimb * prod_{r,c}(dim irr)

                        # decision: optimize for larger edo, go for contracting
                        # block_proj = rtree * ctree
                        rtree = orb_trees[i_new_row, bi]
                        rtree = rtree.reshape(-1, rtree.shape[-1])
                        ctree = ocb_trees[i_new_col, bi].reshape(-1, rtree.shape[1])
                        block_ele_proj = rtree @ ctree.T
                        sh = (rdim, idorb[i_new_row, bi], cdim, idocb[i_new_col, bi])
                        block_ele_proj = block_ele_proj.reshape(sh).swapaxes(1, 2)
                        block_ele_proj = block_ele_proj.reshape(rdim * cdim, idob[bi])

                        # apply CG projector on data
                        ele_dense += block_ele_proj @ data_block

                    # construct elementary block sector in dense tensor
                    sh_ele = ele_degen_dimensions[::-1].ravel()

                    ele_dense = ele_dense.reshape(sh_ele).transpose(reverse_perm)
                    slices = tuple(rslices + cslices)
                    arr[slices] = ele_dense.reshape(ele_degen_dimensions.prod(axis=0))

        assert abs(self.norm() - lg.norm(arr)) <= 1e-13 * self.norm()
        return arr.reshape(self.matrix_shape)

    def _transpose_data(self):
        # optimization is possible but not implemented at the moment
        axes = tuple(range(self._nrr, self._ndim)) + tuple(range(self._nrr))
        return self._permute_data(axes, self._ndim - self._nrr)

    def _permute_data(self, axes, new_nrr):
        si = (1 << np.arange(self._ndim)) @ self._signature
        key = [self._ndim, self._nrr, new_nrr, si, *axes]
        in_reps = self._row_reps + self._col_reps
        for r in in_reps:
            key.append(r.shape[1])
            key.extend(r[1:].ravel())
        key = tuple(key)
        axes = np.array(axes)
        try:
            structural_data = self._structural_data_dic[key]
        except KeyError:
            structural_data = self._compute_structural_data(
                in_reps, self._nrr, axes, new_nrr, self._signature
            )
            self._structural_data_dic[key] = structural_data

        block_irreps, blocks = self._lie_permute_data(axes, new_nrr, structural_data)
        return blocks, block_irreps

    def update_signature(self, sign_update):
        """
        Parameters
        ----------
        sign_update : (ndim,) integer array
            Update to current signature. 0 for no change, 1 for switch in/out, -1 for
            switch in/out with a non-trivial sign change.

        This is an in-place operation.
        """
        # TODO
        raise NotImplementedError("To do!")
        # In the non-abelian case, updating signature can be done from the left or from
        # right, yielding different results.
        up = np.asarray(sign_update, dtype=np.int8)
        if up.shape != (self._ndim,):
            raise ValueError("Invalid shape for sign_update")
        if (np.abs(up) > 2).any():
            raise ValueError("Invalid values in sign_update: must be in [-1, 0, 1]")

        new_sign = self._signature ^ up.astype(bool)

        # only construct projector if needed, ie there is a -1
        # For this case, add diagonal -1 for half integer spins in legs with a loop
        if (up < 0).any():
            # expect uncommon operation: do not store structural_data
            # do not try to optimize, reuse transpose code with a different set of
            # isometries.
            axes = np.arange(self._ndim)
            structural_data = self._compute_structural_data(
                axes, self._nrr, signature_update=up
            )
            degen_data = self._compute_degen_data(axes, self._nrr, structural_data)
            block_irreps, blocks = self._transpose_data(
                axes, self._nrr, structural_data, degen_data
            )

            self._nblocks = len(blocks)
            self._blocks = blocks
            self._block_irreps = block_irreps
        self._signature = new_sign

    def merge_legs(self, i1, i2):
        # TODO
        # merging legs affects the number of elementary blocks: some degeneracies that
        # were seen as "internal" becomes "internal" and end up appearing at a different
        # position in data blocks.
        # It is not enough to just remove one CG tensor at the end of a tree.
        # All involved elementary blocks need to be updated
        raise NotImplementedError("To do!")
