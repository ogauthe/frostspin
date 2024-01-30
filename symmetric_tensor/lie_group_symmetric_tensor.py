import numpy as np
import scipy.linalg as lg
import scipy.sparse as ssp
import numba

from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor


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
    _structural_data_dic = NotImplemented  # permutate information for a given tensor
    _unitary_dic = NotImplemented  # unitaries for a given elementary block

    @classmethod
    def load_isometries(cls, savefile):
        raise NotImplementedError("TODO!")
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

    @classmethod
    def save_isometries(cls, savefile):
        raise NotImplementedError("TODO!")
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

    ####################################################################################
    # Lie group shared symmetry implementation
    ####################################################################################
    @classmethod
    def compute_clebsch_gordan_tree(cls, rep_in, signature, target_irreps=None):
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
    def sliced_elementary_trees(cls, reps, signature, target_irreps=None):
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

    def _compute_structural_data(self, axes, nrr_out, signature_update=None):
        """
        Parameters
        ----------
        axes : 1D integer array
            Permutation as a 1D axis
        nrr_out : integer
            Number of axes considered as row
        signature_update : 1D integer array
            Used to update tensor signature. If provided, axes and nrr_out are not read.

        Returns
        -------
        structural data
        """
        # Purely structural. Can be precomputed.
        # could also be a classmethod with key as input

        in_reps = self._row_reps + self._col_reps
        block_irreps_in, _ = self.get_block_sizes(
            self._row_reps, self._col_reps, self._signature
        )
        nblocks_in = len(block_irreps_in)

        out_signature = self._signature[axes]
        out_row_reps = tuple(in_reps[i] for i in axes[:nrr_out])
        out_col_reps = tuple(in_reps[i] for i in axes[nrr_out:])
        block_irreps_out, _ = self.get_block_sizes(
            out_row_reps, out_col_reps, out_signature
        )
        nblocks_out = len(block_irreps_out)

        elementary_block_per_axis = np.array([r.shape[1] for r in in_reps])
        n_ele_ri = elementary_block_per_axis[: self._nrr].prod()
        n_ele_ci = elementary_block_per_axis[self._nrr :].prod()
        n_ele_ro = elementary_block_per_axis[axes[:nrr_out]].prod()
        n_ele_co = elementary_block_per_axis[axes[nrr_out:]].prod()

        idirb = np.zeros((n_ele_ri, nblocks_in), dtype=int)
        idicb = np.zeros((n_ele_ci, nblocks_in), dtype=int)
        idorb = np.zeros((n_ele_ro, nblocks_out), dtype=int)
        idocb = np.zeros((n_ele_co, nblocks_out), dtype=int)
        cg_trees_ri = [None] * n_ele_ri
        cg_trees_ci = [None] * n_ele_ci
        cg_trees_ro = [None] * n_ele_ro
        cg_trees_co = [None] * n_ele_co
        ele_indices = []

        key0 = np.empty((2 * self._ndim + 3,), dtype=np.int64)
        key0[self._ndim] = self._nrr
        key0[self._ndim + 1] = nrr_out
        key0[self._ndim + 2] = (1 << np.arange(self._ndim)) @ self._signature
        key0[self._ndim + 3 :] = axes
        singlet = self.singlet[1, 0]

        # convention: return elementary unitary blocked sliced for IN irrep blocks
        isometry_blocks = []

        # need to add idirb and idicb axes in permutation
        perm = np.empty((self._ndim + 2,), dtype=int)
        perm[: self._ndim] = (axes >= self._nrr) + axes
        perm[self._ndim] = self._nrr
        perm[self._ndim + 1] = self._ndim + 1

        # strides to recover ir/ic_in/out from i_ele
        strides = np.zeros((4, self._ndim), dtype=int)
        strides[0, self._nrr - 1] = 1
        for i in reversed(range(1, self._nrr)):
            strides[0, i - 1] = elementary_block_per_axis[i] * strides[0, i]
        strides[1, self._ndim - 1] = 1
        for i in reversed(range(self._nrr + 1, self._ndim)):
            strides[1, i - 1] = elementary_block_per_axis[i] * strides[1, i]
        strides[2, axes[nrr_out - 1]] = 1
        for i in reversed(range(1, nrr_out)):
            strides[2, axes[i - 1]] = (
                elementary_block_per_axis[axes[i]] * strides[2, axes[i]]
            )
        strides[3, axes[self._ndim - 1]] = 1
        for i in reversed(range(nrr_out + 1, self._ndim)):
            strides[3, axes[i - 1]] = (
                elementary_block_per_axis[axes[i]] * strides[3, axes[i]]
            )

        sig_ri = self._signature[: self._nrr]
        sig_ci = ~self._signature[self._nrr :]
        sig_ro = self._signature[axes[:nrr_out]]
        sig_co = ~self._signature[axes[nrr_out:]]

        for i_ele in range(n_ele_ri * n_ele_ci):
            i_mul = np.array(np.unravel_index(i_ele, elementary_block_per_axis))
            reps_ei = [None] * self._ndim
            for i in range(self._ndim):
                irr = in_reps[i][1, i_mul[i]]
                reps_ei[i] = np.array([[1], [irr]])
                key0[i] = irr
            ri_rep = self.combine_representations(reps_ei[: self._nrr], sig_ri)
            ci_rep = self.combine_representations(reps_ei[self._nrr :], sig_ci)
            shared_ri = (ri_rep[1, :, None] == ci_rep[1]).nonzero()[0]

            if shared_ri.size:  # if the block contributes
                ele_inds = strides @ i_mul
                ir_in, ic_in, ir_out, ic_out = ele_inds
                ele_indices.append(ele_inds)
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
                    ro_rep = self.combine_representations(reps_ero, sig_ro)
                    inds_e, inds_f = (ro_rep[1, :, None] == block_irreps_out).nonzero()
                    idorb[ir_out, inds_f] = ro_rep[0, inds_e]
                if not idocb[ic_out].any():
                    co_rep = self.combine_representations(reps_eco, sig_co)
                    inds_e, inds_f = (co_rep[1, :, None] == block_irreps_out).nonzero()
                    idocb[ic_out, inds_f] = co_rep[0, inds_e]

                # prune singlets
                kept = (key0[: self._ndim] != singlet).nonzero()[0]
                if kept.size == self._ndim:
                    key = tuple(key0)
                else:
                    nrr_kept = (kept < self._nrr).sum()
                    sig_kept = (1 << np.arange(kept.size)) @ self._signature[kept]
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
                    ele_unitary = self._unitary_dic[key]
                except KeyError:
                    # we need to use block_irreps_in as target irreps, not ele_irreps_in
                    # as some block may be absent in ic_in, but not for ic_in + 1
                    if cg_trees_ri[ir_in] is None:
                        idirb_e, trees = self.sliced_elementary_trees(
                            reps_ei[: self._nrr],
                            sig_ri,
                            target_irreps=block_irreps_in,
                        )
                        cg_trees_ri[ir_in] = trees[0]  # only 1 ele block
                        assert (idirb_e == idirb[ir_in]).all()
                    if cg_trees_ci[ic_in] is None:
                        idicb_e, trees = self.sliced_elementary_trees(
                            reps_ei[self._nrr :],
                            sig_ci,
                            target_irreps=block_irreps_in,
                        )
                        cg_trees_ci[ic_in] = trees[0]
                        assert (idicb_e == idicb[ic_in]).all()
                    if cg_trees_ro[ir_out] is None:
                        idorb_e, trees = self.sliced_elementary_trees(
                            reps_ero,
                            sig_ro,
                            target_irreps=block_irreps_out,
                        )
                        cg_trees_ro[ir_out] = trees[0]
                        assert (idorb_e == idorb[ir_out]).all()
                    if cg_trees_co[ic_out] is None:
                        idocb_e, trees = self.sliced_elementary_trees(
                            reps_eco,
                            sig_co,
                            target_irreps=block_irreps_out,
                        )
                        assert (idocb_e == idocb[ic_out]).all()
                        cg_trees_co[ic_out] = trees[0]

                    ele_unitary = self.overlap_cg_trees(
                        cg_trees_ri[ir_in],
                        cg_trees_ci[ic_in],
                        cg_trees_ro[ir_out],
                        cg_trees_co[ic_out],
                        perm,
                    )
                    # save elementary block unitary for later
                    self._unitary_dic[key] = ele_unitary

                assert idirb[ir_in] @ idicb[ic_in] == idorb[ir_out] @ idocb[ic_out]
                # map ele in_blocks_index to full tensor in_blocks_index
                unitary_blocks = [None] * nblocks_in
                for i, eub in enumerate(ele_unitary):
                    bi_in = ele_in_binds[i]
                    unitary_blocks[bi_in] = eub
                isometry_blocks.append(unitary_blocks)

        ele_indices = np.array(ele_indices)
        isometry_in_blocks = np.empty((ele_indices.shape[0], nblocks_in), dtype=object)
        isometry_in_blocks[:] = isometry_blocks
        assert (
            (idirb[ele_indices[:, 0]] * idicb[ele_indices[:, 1]]).sum(axis=1)
            == (idorb[ele_indices[:, 2]] * idocb[ele_indices[:, 3]]).sum(axis=1)
        ).all(), "number of coefficient not conserved"

        structual_data = (
            ele_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            block_irreps_out,
            isometry_in_blocks,
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
        for rtree, ctree in zip(cg_ro, cg_co):
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
                ele_unitary_blocks.append(isometry_bi)

        return ele_unitary_blocks

    def _transpose_data(self, axes, nrr_out, structural_data):
        """
        Move data and construct new data blocks
        """
        # 3 nested loops: elementary blocks, rows, columns
        # change loop order?

        in_reps = self._row_reps + self._col_reps
        out_row_reps = tuple(in_reps[i] for i in axes[:nrr_out])
        out_col_reps = tuple(in_reps[i] for i in axes[nrr_out:])

        (
            ele_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            block_irreps_out,
            isometry_in_blocks,
        ) = structural_data

        external_degen_ir = _numba_compute_external_degen(self._row_reps)
        external_degen_ic = _numba_compute_external_degen(self._col_reps)
        external_degen_or = _numba_compute_external_degen(out_row_reps)
        external_degen_oc = _numba_compute_external_degen(out_col_reps)

        edir = external_degen_ir.prod(axis=1)
        edic = external_degen_ic.prod(axis=1)
        edor = external_degen_or.prod(axis=1)
        edoc = external_degen_oc.prod(axis=1)

        nblocks_out = len(block_irreps_out)
        assert idorb.shape[1] == nblocks_out
        assert idocb.shape[1] == nblocks_out
        assert external_degen_or.shape[0] == idorb.shape[0]
        assert external_degen_oc.shape[0] == idocb.shape[0]
        assert external_degen_ir.shape[0] == idirb.shape[0]
        assert external_degen_ic.shape[0] == idicb.shape[0]

        assert self._nblocks <= idirb.shape[1]
        if self._nblocks < idirb.shape[1]:  # if a block is missing
            # filter out missing blocks
            block_irreps_in, _ = self.get_block_sizes(
                self._row_reps, self._col_reps, self._signature
            )
            _, block_inds = (self._block_irreps[:, None] == block_irreps_in).nonzero()
            # possible also filter ele_indices with (idirb.T @ idicb).nonzero()[0]
            # probably not worth it
        else:
            block_inds = range(self._nblocks)

        slices_ir = (edir[:, None] * idirb).cumsum(axis=0)
        slices_ic = (edic[:, None] * idicb).cumsum(axis=0)
        slices_or = (edor[:, None] * idorb).cumsum(axis=0)
        slices_oc = (edoc[:, None] * idocb).cumsum(axis=0)

        # need to initalize blocks_out in case of missing blocks_in
        rshb = slices_or[-1]
        cshb = slices_oc[-1]
        dtype = self.dtype
        blocks_out = tuple(np.zeros(sh, dtype=dtype) for sh in zip(rshb, cshb))

        data_perm = (0, self._nrr + 1, *(axes + 1 + (axes >= self._nrr)))

        for i_ele, indices in enumerate(ele_indices):
            # edor = external degeneracy out row
            # idicb = internal degeneracy in column block

            i_ir, i_ic, i_or, i_oc = indices
            edir_ele = edir[i_ir]
            edic_ele = edic[i_ic]
            ele_sh = [0, *external_degen_ir[i_ir], 0, *external_degen_ic[i_ic]]
            edor_ele = edor[i_or]
            edoc_ele = edoc[i_oc]
            ed = edor_ele * edoc_ele
            assert ed == edir_ele * edic_ele

            out_data = np.zeros((idorb[i_or] @ idocb[i_oc], ed), dtype=dtype)

            for ib_self, ibi in enumerate(block_inds):  # missing blocks are filtered
                idib = idirb[i_ir, ibi] * idicb[i_ic, ibi]
                # need to check if this block_irrep_in appears in elementary_block
                if idib > 0:
                    assert isometry_in_blocks[i_ele, ibi] is not None
                    sir2 = slices_ir[i_ir, ibi]
                    sir1 = sir2 - edir_ele * idirb[i_ir, ibi]
                    sic2 = slices_ic[i_ic, ibi]
                    sic1 = sic2 - edic_ele * idicb[i_ic, ibi]

                    # there are two operations: changing basis with elementary AND
                    # swapping axes in external degeneracy part. Transpose tensor BEFORE
                    # applying unitary to do only one transpose

                    in_data = self._blocks[ib_self][sir1:sir2, sic1:sic2]

                    # initial tensor shape = (
                    # internal degeneracy in row block,
                    # *external degeneracies per in row axes,
                    # internal degeneracy in col block,
                    # *external degeneracies per in col axes)
                    ele_sh[0] = idirb[i_ir, ibi]
                    ele_sh[self._nrr + 1] = idicb[i_ic, ibi]
                    in_data = in_data.reshape(ele_sh)

                    # transpose to shape = (
                    # internal degeneracy in row block,
                    # internal degeneracy in col, block,
                    # *external degeneracies per OUT row axes,
                    # *external degeneracies per OUT col axes)
                    in_data = in_data.transpose(data_perm).reshape(idib, ed)

                    # convention: iso_iblock is sliced irrep-wise on its columns = "IN"
                    # but not for its rows = "OUT"
                    # meaning it is applied to "IN" irrep block data, but generates data
                    # for all OUT irrep blocks
                    out_data += isometry_in_blocks[i_ele, ibi] @ in_data

            # transpose out_data only after every IN blocks have been processed
            oshift = 0
            for ibo in range(nblocks_out):
                idorb_ele = idorb[i_or, ibo]
                idocb_ele = idocb[i_oc, ibo]
                idob = idorb_ele * idocb_ele

                if idob > 0:
                    out_block = out_data[oshift : oshift + idob]
                    sh = (idorb_ele, idocb_ele, edor_ele, edoc_ele)
                    out_block = out_block.reshape(sh).swapaxes(1, 2)
                    sh2 = (edor_ele * idorb_ele, edoc_ele * idocb_ele)
                    out_block = out_block.reshape(sh2)

                    # different IN block irreps may contribute: need +=
                    sor2 = slices_or[i_or, ibo]
                    sor1 = sor2 - edor_ele * idorb_ele
                    soc2 = slices_oc[i_oc, ibo]
                    soc1 = soc2 - edoc_ele * idocb_ele
                    blocks_out[ibo][sor1:sor2, soc1:soc2] = out_block
                    oshift += idob

        return block_irreps_out, blocks_out

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################

    @classmethod
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        # require that arr has structure [degen1*irrep1, degen2*irrep2 ...]
        # e.g. (singlet1, singlet2, up1, down1, up2, down2)

        row_reps = tuple(row_reps)
        col_reps = tuple(col_reps)
        in_reps = row_reps + col_reps
        if arr.shape != tuple(cls.representation_dimension(rep) for rep in in_reps):
            raise ValueError("Representations do not match array shape")

        nrr = len(row_reps)
        ndim = len(in_reps)
        if signature is None:
            signature = np.arange(ndim) >= nrr
        else:
            signature = np.ascontiguousarray(signature, dtype=bool)
            if signature.shape != (arr.ndim,):
                raise ValueError("Signature does not match array shape")

        block_irreps, block_shapes_in = cls.get_block_sizes(
            row_reps, col_reps, signature
        )

        # define shifts to localize elementary block position in dense format
        rshifts = []
        elementary_block_rows = np.empty((nrr,), dtype=int)
        for i, rep in enumerate(row_reps):
            s = [0]
            elementary_block_rows[i] = rep.shape[1]
            for j in range(rep.shape[1]):
                s.append(cls.representation_dimension(rep[:, : j + 1]))
            rshifts.append(s)

        cshifts = []
        elementary_block_cols = np.empty((ndim - nrr,), dtype=int)
        for i, rep in enumerate(col_reps):
            s = [0]
            elementary_block_cols[i] = rep.shape[1]
            for j in range(rep.shape[1]):
                s.append(cls.representation_dimension(rep[:, : j + 1]))
            cshifts.append(s)

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
        external_degen_ir = _numba_compute_external_degen(row_reps)
        external_degen_ic = _numba_compute_external_degen(col_reps)
        edir = external_degen_ir.prod(axis=1)
        edic = external_degen_ic.prod(axis=1)
        block_shifts_row = (edir[:, None] * idirb).cumsum(axis=0)
        block_shifts_col = (edic[:, None] * idicb).cumsum(axis=0)

        # initialize blocks sizes
        blocks = tuple(np.empty(sh, dtype=arr.dtype) for sh in block_shapes_in)
        irrep_dims = [cls.irrep_dimension(irr) for irr in block_irreps]

        # loop over row elementary blocks
        for i_ir in range(idirb.shape[0]):
            ele_degen_dimensions = np.empty((2, ndim), dtype=int)
            ele_degen_dimensions[0, :nrr] = external_degen_ir[i_ir]
            irmul = np.unravel_index(i_ir, elementary_block_rows)
            rslices = []
            for i in range(nrr):
                ele_degen_dimensions[1, i] = cls.irrep_dimension(
                    row_reps[i][1, irmul[i]]
                )
                rslices.append(slice(rshifts[i][irmul[i]], rshifts[i][irmul[i] + 1]))
            rdim = ele_degen_dimensions[1, :nrr].prod()

            # loop over column elementary blocks
            for i_ic in range(idicb.shape[0]):
                idi = idirb[i_ir].T @ idicb[i_ic]
                if idi > 0:
                    ele_degen_dimensions[0, nrr:] = external_degen_ic[i_ic]
                    icmul = np.unravel_index(i_ic, elementary_block_cols)
                    cslices = []
                    for i in range(ndim - nrr):
                        ele_degen_dimensions[1, nrr + i] = cls.irrep_dimension(
                            col_reps[i][1, icmul[i]]
                        )
                        cslices.append(
                            slice(cshifts[i][icmul[i]], cshifts[i][icmul[i] + 1])
                        )
                    edi = edir[i_ir] * edic[i_ic]
                    cdim = ele_degen_dimensions[1, nrr:].prod()

                    # split degen and structural
                    slices = tuple(rslices + cslices)
                    sh = ele_degen_dimensions.T.ravel()
                    data = arr[slices].reshape(sh)
                    assert rdim * cdim == ele_degen_dimensions[1].prod()
                    data = data.transpose(degen_irrep_perm).reshape(rdim * cdim, edi)
                    idib = idirb[i_ir] * idicb[i_ic]
                    for bi in idib.nonzero()[0]:
                        # construct CG projector on block irrep elementary sector
                        # see toarray fro a discussion of the contraction scheme

                        rtree = irb_trees[i_ir, bi]
                        rtree = rtree.reshape(-1, rtree.shape[-1])
                        ctree = icb_trees[i_ic, bi].reshape(-1, rtree.shape[1])
                        block_ele_proj = rtree @ ctree.T
                        sh = (rdim, idirb[i_ir, bi], cdim, idicb[i_ic, bi])
                        block_ele_proj = block_ele_proj.reshape(sh).swapaxes(1, 2)
                        block_ele_proj = block_ele_proj.reshape(rdim * cdim, idib[bi])

                        data_block = block_ele_proj.T @ data

                        # data_block still has a non-trivial structure due to inner
                        # degeneracies. Its shape is
                        # (int_degen_row, int_degen_col, ext_degen_row, ext_degen_col)
                        # we need to permute axes to reshape as
                        # (ext_degen_row * int_degen_row, ext_degen_col * int_degen_col)
                        sh = (idirb[i_ir, bi], idicb[i_ic, bi], edir[i_ir], edic[i_ic])
                        sh2 = (sh[0] * sh[2], sh[1] * sh[3])
                        data_block = data_block.reshape(sh).swapaxes(1, 2)
                        data_block = (data_block / irrep_dims[bi]).reshape(sh2)

                        rs = slice(
                            block_shifts_row[i_ir, bi] - sh2[0],
                            block_shifts_row[i_ir, bi],
                        )
                        cs = slice(
                            block_shifts_col[i_ic, bi] - sh2[1],
                            block_shifts_col[i_ic, bi],
                        )
                        blocks[bi][rs, cs] = data_block

        st = cls(row_reps, col_reps, blocks, block_irreps, signature)
        assert abs(st.norm() - lg.norm(arr)) <= 1e-13 * lg.norm(arr)
        return st

    def toarray(self, as_matrix=False):
        # define shifts to localize elementary block position in dense format
        rshifts = []
        elementary_block_rows = np.empty((self._nrr,), dtype=int)
        for i, rep in enumerate(self._row_reps):
            s = [0]
            elementary_block_rows[i] = rep.shape[1]
            for j in range(rep.shape[1]):
                s.append(self.representation_dimension(rep[:, : j + 1]))
            rshifts.append(s)

        cshifts = []
        elementary_block_cols = np.empty((self._ndim - self._nrr,), dtype=int)
        for i, rep in enumerate(self._col_reps):
            s = [0]
            elementary_block_cols[i] = rep.shape[1]
            for j in range(rep.shape[1]):
                s.append(self.representation_dimension(rep[:, : j + 1]))
            cshifts.append(s)

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
        for i_or in range(idorb.shape[0]):
            ele_degen_dimensions = np.empty((2, self._ndim), dtype=int)
            ele_degen_dimensions[0, : self._nrr] = external_degen_or[i_or]
            irmul = np.unravel_index(i_or, elementary_block_rows)
            rslices = []
            for i in range(self._nrr):
                ele_degen_dimensions[1, i] = self.irrep_dimension(
                    self._row_reps[i][1, irmul[i]]
                )
                rslices.append(slice(rshifts[i][irmul[i]], rshifts[i][irmul[i] + 1]))
            rdim = ele_degen_dimensions[1, : self._nrr].prod()

            # loop over column elementary blocks
            for i_oc in range(idocb.shape[0]):
                ido = idorb[i_or].T @ idocb[i_oc]
                if ido > 0:  # if this elementary block is allowed
                    ele_degen_dimensions[0, self._nrr :] = external_degen_oc[i_oc]
                    icmul = np.unravel_index(i_oc, elementary_block_cols)
                    cslices = []
                    for i in range(self._ndim - self._nrr):
                        ele_degen_dimensions[1, self._nrr + i] = self.irrep_dimension(
                            self._col_reps[i][1, icmul[i]]
                        )
                        cslices.append(
                            slice(cshifts[i][icmul[i]], cshifts[i][icmul[i] + 1])
                        )
                    edo = edor[i_or] * edoc[i_oc]
                    cdim = ele_degen_dimensions[1, self._nrr :].prod()
                    ele_dense = np.zeros((rdim * cdim, edo), dtype=dtype)
                    idob = idorb[i_or] * idocb[i_oc]
                    for bi in idob.nonzero()[0]:
                        rs = slice(
                            block_shifts_row[i_or, bi] - edor[i_or] * idorb[i_or, bi],
                            block_shifts_row[i_or, bi],
                            1,
                        )
                        cs = slice(
                            block_shifts_col[i_oc, bi] - edoc[i_oc] * idocb[i_oc, bi],
                            block_shifts_col[i_oc, bi],
                            1,
                        )
                        sh = (idorb[i_or, bi], edor[i_or], idocb[i_oc, bi], edoc[i_oc])
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
                        #  1    2   3 ...           1    2  ...
                        #
                        #
                        # 2 ways of contracting this:
                        # data_block * (rtree * ctree), with complexity
                        #    prod_{r,c}(dim irr) * dim * ido
                        #  + edo * ido * prod_{r,c}(dim irr)
                        # or (data_block * rtree) * ctree with complexity
                        #    prod_{r}(dim irr) * dim * edo * ido
                        #  + edo * idocb * diim * prod_{r,c}(dim irr)

                        # decision: optimize for larger edo, go for contracting
                        # block_proj = rtree * ctree
                        rtree = orb_trees[i_or, bi]
                        rtree = rtree.reshape(-1, rtree.shape[-1])
                        ctree = ocb_trees[i_oc, bi].reshape(-1, rtree.shape[1])
                        block_ele_proj = rtree @ ctree.T
                        sh = (rdim, idorb[i_or, bi], cdim, idocb[i_oc, bi])
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
        if as_matrix:
            return arr.reshape(self.matrix_shape)
        return arr

    @property
    def T(self):
        row_axes = tuple(range(self._nrr, self._ndim))
        col_axes = tuple(range(self._nrr))
        return self.permutate(row_axes, col_axes)

    def permutate(self, row_axes, col_axes):
        axes = np.concatenate((row_axes, col_axes))
        nrr_out = len(row_axes)
        trivial = np.arange(self._ndim)

        if axes.shape != (self._ndim,) or (np.sort(axes) != trivial).any():
            raise ValueError("axes do not match tensor")

        # return early for identity only, matrix transpose is not trivial
        if nrr_out == self._nrr and (axes == trivial).all():
            return self

        si = (1 << np.arange(self._ndim)) @ self._signature
        key = [self._ndim, self._nrr, nrr_out, si, *axes]
        for r in self._row_reps + self._col_reps:
            key.append(r.shape[1])
            key.extend(r[1:].ravel())
        key = tuple(key)
        try:
            structural_data = self._structural_data_dic[key]
        except KeyError:
            structural_data = self._compute_structural_data(axes, nrr_out)
            self._structural_data_dic[key] = structural_data

        block_irreps, blocks = self._transpose_data(axes, nrr_out, structural_data)

        reps = []
        for ax in axes:
            if ax < self._nrr:
                reps.append(self._row_reps[ax])
            else:
                reps.append(self._col_reps[ax - self._nrr])
        signature = self._signature[axes]
        ret = type(self)(
            reps[:nrr_out], reps[nrr_out:], blocks, block_irreps, signature
        )
        assert abs(ret.norm() - self.norm()) <= 1e-13 * self.norm()
        return ret

    def update_signature(self, sign_update):
        """
        Parameters
        ----------
        sign_update : (ndim,) integer array
            Update to current signature. 0 for no change, 1 for switch in/out, -1 for
            switch in/out with a non-trivial sign change.

        This is an in-place operation.
        """
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
        return

    def merge_legs(self, i1, i2):
        # TODO
        # merging legs affects the number of elementary blocks: some degeneracies that
        # were seen as "internal" becomes "internal" and end up appearing at a different
        # position in data blocks.
        # It is not enough to just remove one CG tensor at the end of a tree.
        # All involved elementary blocks need to be updated
        raise NotImplementedError("To do!")
