import numpy as np
import scipy.linalg as lg
import scipy.sparse as ssp

from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor


class LieGroupSymmetricTensor(NonAbelianSymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with non-abelian symmetry defined
    by a Lie group. Axis permutation is done using isometries defined by fusion trees of
    representations.
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    # every method is group-specific

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    _structural_data_dic = NotImplemented

    @classmethod
    def construct_matrix_projector(cls, row_reps, col_reps, signature):
        # TODO: rewrite it as
        # construct_singlet_space_projector(cls, irreps, signature, nrr)
        r"""
                    singlet space
                    /          \
                   /            \
                prod_l        prod_r
                 /               /
                /\              /\
               /\ \            /\ \
             row_reps        col_reps

        Parameters
        ----------
        row_reps : tuple of ndarray
            Row axis representations.
        col_reps : tuple of ndarray
            Column axis representations.
        signature : 1D bool array or None
            Leg signatures.

        Returns
        -------
        proj : (M, N) sparse matrix
            Projector on singlet, with M the dimension of the full parameter space and
            N the singlet space dimension.
        """
        raise NotImplementedError("Must be defined in derived class")

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

    def _compute_elementary_transpose(
        self, ele_r_in, ele_c_in, ele_r_out, ele_c_out, perm, out_signature
    ):
        p_in = self.construct_matrix_projector(ele_r_in, ele_c_in, self._signature)
        p_out = self.construct_matrix_projector(ele_r_out, ele_c_out, out_signature)
        assert p_in.shape == p_out.shape

        in_sh_ele = tuple(self.irrep_dimension(r[1, 0]) for r in ele_r_in + ele_c_in)
        p_in = p_in.reshape(in_sh_ele + (p_in.shape[1],))
        p_in = p_in.transpose(perm)
        p_in = p_in.reshape(p_out.shape)
        unitary_ele = p_out.T @ p_in
        return unitary_ele

    def _compute_elementary_conjugate(self, ele_r_in, ele_c_in, signature_update):
        p_in = self.construct_matrix_projector(ele_r_in, ele_c_in, self._signature)
        mid = ssp.eye(np.eye(1), format="coo")
        for i, r in enumerate(ele_r_in + ele_c_in):
            dim = self.representation_dimension(r)
            if signature_update[i] == -1:
                z = self.construct_matrix_projector([r], [r], self._signature[[i, i]])
                z = np.sqrt(dim) * z.reshape(dim, dim)
                zrep = z @ z
            else:
                zrep = ssp.eye(dim, format="coo")
            mid = ssp.kron(mid, zrep, format="coo")

        unitary_ele = p_in.T @ mid @ p_in
        return unitary_ele

    @classmethod
    def sliced_elementary_trees(cls, reps, signature, target_irreps=None):
        """
        Construct Clebsch-Gordon tree for all elementary blocks

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
        if target_irreps is None:
            total_rep = cls.combine_representations(reps, signature)
            target_irreps = total_rep[1]

        elementary_block_per_axis = np.array([r.shape[1] for r in reps])
        n_ele_blocks = elementary_block_per_axis.prod()
        n_irreps = len(target_irreps)
        internal_degeneracies = np.zeros((n_ele_blocks, n_irreps), dtype=int)
        ele_trees = np.empty((n_ele_blocks, n_irreps), dtype=object)
        max_irrep = target_irreps[-1]

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
                ele_reps, signature, max_irrep=max_irrep
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

        block_irreps_in, _ = self.get_block_sizes(
            self._row_reps, self._col_reps, self._signature
        )
        nblocks_in = len(block_irreps_in)

        out_signature = self._signature[axes]
        in_reps = self._row_reps + self._col_reps
        out_row_reps = tuple(in_reps[i] for i in axes[:nrr_out])
        out_col_reps = tuple(in_reps[i] for i in axes[nrr_out:])
        block_irreps_out, _ = self.get_block_sizes(
            out_row_reps, out_col_reps, out_signature
        )
        nblocks_out = len(block_irreps_out)

        # precompute CG trees
        # a few ones may never be used (think elementary row block with only
        # half-integer spin row when all elementary column blocks are integer spins)
        # do not bother pruning them, max_irrep already limits tree cost
        idirb, irb_trees = self.sliced_elementary_trees(
            self._row_reps, self._signature[: self._nrr], target_irreps=block_irreps_in
        )
        idicb, icb_trees = self.sliced_elementary_trees(
            self._col_reps, ~self._signature[self._nrr :], target_irreps=block_irreps_in
        )
        idorb, orb_trees = self.sliced_elementary_trees(
            out_row_reps, out_signature[:nrr_out], target_irreps=block_irreps_out
        )
        idocb, ocb_trees = self.sliced_elementary_trees(
            out_col_reps, ~out_signature[nrr_out:], target_irreps=block_irreps_out
        )
        assert (idirb @ idicb.T).sum() == (idorb @ idocb.T).sum()  # coefficient number

        # indices of elementary blocks authorized by symmetry
        contribute_ele = (idirb @ idicb.T).ravel().nonzero()[0]
        n_ele = contribute_ele.size

        # we need to find the row and col indices ot these elementary blocks in OUT
        elementary_block_per_axis = np.array([r.shape[1] for r in in_reps])
        ncol_blocks_in = elementary_block_per_axis[self._nrr :].prod()
        ele_indices = np.empty((4, n_ele), dtype=int)
        ele_indices[:2] = np.divmod(contribute_ele, ncol_blocks_in)
        mul = np.array(np.unravel_index(contribute_ele, elementary_block_per_axis))
        strides = np.zeros((2, self._ndim), dtype=int)
        s = 1
        for i in reversed(range(nrr_out)):
            strides[0, axes[i]] = s
            s *= elementary_block_per_axis[axes[i]]
        s = 1
        for i in reversed(range(nrr_out, self._ndim)):
            strides[1, axes[i]] = s
            s *= elementary_block_per_axis[axes[i]]
        ele_indices[2:] = strides @ mul
        ele_indices = ele_indices.T.copy()

        # normalizing factor
        dims_out = [self.irrep_dimension(irr) for irr in block_irreps_out]

        # convention: return elementary unitary blocked sliced for IN irrep blocks
        isometry_in_blocks = np.empty((n_ele, nblocks_in), dtype=object)

        # need to add idirb and idicb axes in permutation
        perm = np.empty((self._ndim + 2,), dtype=int)
        perm[self._ndim + 1] = self._ndim + 1
        perm[self._ndim] = self._nrr
        perm[: self._ndim] = axes
        perm[: self._ndim][axes >= self._nrr] += 1

        for i_ele in range(n_ele):
            ir_in, ic_in, ir_out, ic_out = ele_indices[i_ele]

            # precompute OUT singlet projector for all OUT block irreps that appear
            out_block_indices = (idorb[ir_out] * idocb[ic_out]).nonzero()[0]
            out_proj_block = [None] * nblocks_out
            for bi in out_block_indices:
                rtree = orb_trees[ir_out, bi]  # shape (*dim_ele_r, idorb, irr)
                ctree = ocb_trees[ic_out, bi]  # shape (*dim_ele_c, idocb, irr)
                dim = rtree.shape[-1]  # dim(irrep)
                rt = rtree.reshape(-1, dim)  # shape (dim_ele_r * idorb, irrep_dim)
                ct = ctree.reshape(-1, dim)  # shape (irrep_dim, dim_ele_c * idocb)
                block_proj = rt @ ct.T  # shape (dim_ele_r * idorb, dim_ele_c * idocb)
                sh = (
                    rt.shape[0] // rtree.shape[-2],
                    rtree.shape[-2],
                    ct.shape[0] // ctree.shape[-2],
                    ctree.shape[-2],
                )
                block_proj = block_proj.reshape(sh).swapaxes(1, 2)
                nsh = (sh[0] * sh[2], sh[1] * sh[3])  # shape (dim_ele, idorb * idocb)
                out_proj_block[bi] = block_proj.reshape(nsh).T

            for bi_in in (idirb[ir_in] * idicb[ic_in]).nonzero()[0]:
                rtree = irb_trees[ir_in, bi_in]
                ctree = icb_trees[ic_in, bi_in]
                irrep_dim = rtree.shape[-1]
                rt = rtree.reshape(-1, irrep_dim)
                ct = ctree.reshape(-1, irrep_dim)
                block_proj = rt @ ct.T  # shape (dim_ele_r * idirb, dim_ele_c * idicb)
                sh = rtree.shape[:-1] + ctree.shape[:-1]
                swapped = block_proj.reshape(sh).transpose(perm)
                swapped = swapped.reshape(-1, rtree.shape[-2] * ctree.shape[-2])

                sh = (
                    idorb[ir_out] @ idocb[ic_out],
                    idirb[ir_in, bi_in] * idicb[ic_in, bi_in],
                )
                isometry_bi = np.empty(sh)

                k = 0
                for bi_out in out_block_indices:
                    d = idorb[ir_out, bi_out] * idocb[ic_out, bi_out]
                    isometry = out_proj_block[bi_out] @ swapped
                    # not a unitary: not square + sqrt(irrep_dim) not included
                    # with sqrt(irrep) and packed with matrices from other irrep blocks
                    # it becomes unitary => coefficients are of order 1, it is safe to
                    # delete numerical zeros according to absolute tolerance
                    isometry[np.abs(isometry) < 1e-14] = 0
                    isometry_bi[k : k + d] = isometry / dims_out[bi_out]
                    k += d

                assert k == isometry_bi.shape[0]
                isometry_in_blocks[i_ele, bi_in] = isometry_bi

        structual_data = (
            ele_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            isometry_in_blocks,
        )
        return structual_data

    def _compute_degen_data(self, axes, nrr_out, structural_data):
        """
        Compute indices and slices, depending on both internal external degeneracies.
        Symmetry-agnostic, but depends on self degeneracies. Data is not accessed.
        """
        # should be possible to compile it
        (
            ele_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            _,
        ) = structural_data
        in_reps = self._row_reps + self._col_reps
        out_row_reps = [in_reps[ax] for ax in axes[:nrr_out]]
        out_col_reps = [in_reps[ax] for ax in axes[nrr_out:]]

        def compute_external_degen(reps):
            nr = len(reps)
            strides = np.array([r.shape[1] for r in reps])
            n_ele_blocks = strides.prod()
            mul_indices = np.unravel_index(np.arange(n_ele_blocks), strides)
            external_degen = np.empty((nr, n_ele_blocks), dtype=int)
            for i in range(nr):
                external_degen[i] = reps[i][0, mul_indices[i]]
            return external_degen

        external_degen_ir = compute_external_degen(self._row_reps)
        external_degen_ic = compute_external_degen(self._col_reps)
        external_degen_or = compute_external_degen(out_row_reps)
        external_degen_oc = compute_external_degen(out_col_reps)

        edir = external_degen_ir.prod(axis=0)
        edic = external_degen_ic.prod(axis=0)
        edor = external_degen_or.prod(axis=0)
        edoc = external_degen_oc.prod(axis=0)

        slices_ir = (edir[:, None] * idirb).cumsum(axis=0)
        slices_ic = (edic[:, None] * idicb).cumsum(axis=0)
        slices_or = (edor[:, None] * idorb).cumsum(axis=0)
        slices_oc = (edoc[:, None] * idocb).cumsum(axis=0)

        degen_data = (
            external_degen_ir.T.copy(),
            external_degen_ic.T.copy(),
            edir,
            edic,
            edor,
            edoc,
            slices_ir,
            slices_ic,
            slices_or,
            slices_oc,
        )
        return degen_data

    def _transpose_data(self, axes, nrr_out, structural_data, degen_data):
        """
        Move data and construct new data blocks
        """
        # 3 nested loops: elementary blocks, rows, columns
        # change loop order?

        (
            ele_indices,
            idirb,
            idicb,
            idorb,
            idocb,
            isometry_in_blocks,
        ) = structural_data
        (
            external_degen_ir,
            external_degen_ic,
            edir,
            edic,
            edor,
            edoc,
            slices_ir,
            slices_ic,
            slices_or,
            slices_oc,
        ) = degen_data

        in_reps = self._row_reps + self._col_reps
        out_reps = tuple(in_reps[i] for i in axes)

        # need to initalize blocks_out in case of missing blocks_in
        block_irreps_out, block_shapes_out = self.get_block_sizes(
            out_reps[:nrr_out], out_reps[nrr_out:], self._signature[axes]
        )
        blocks_out = tuple(np.zeros(sh) for sh in block_shapes_out)
        nblocks_out = len(blocks_out)
        block_irreps_in, _ = self.get_block_sizes(
            self._row_reps, self._col_reps, self._signature
        )

        data_perm = tuple(ax + 1 if ax < self._nrr else ax + 2 for ax in axes)
        data_perm = (0, self._nrr + 1) + data_perm

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

            out_data = np.zeros((idirb[i_ir] @ idicb[i_ic], ed))

            for ibi, irr in enumerate(block_irreps_in):
                # need to check if this block_irrep_in appears in elementary_block
                # AND if the block exists in tensor
                idib = idirb[i_ir, ibi] * idicb[i_ic, ibi]
                ib_self = self._block_irreps.searchsorted(irr)
                if (
                    idib > 0
                    and ib_self < self._nblocks
                    and self._block_irreps[ib_self] == irr
                ):
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
                    sor1 = sor2 - edor[i_or] * idorb[i_or, ibo]
                    soc2 = slices_oc[i_oc, ibo]
                    soc1 = soc2 - edoc[i_oc] * idocb[i_oc, ibo]
                    assert sh2 == (sor2 - sor1, soc2 - soc1)
                    blocks_out[ibo][sor1:sor2, soc1:soc2] = out_block
                    oshift += idob
            assert oshift == idirb[i_ir] @ idicb[i_ic]

        return block_irreps_out, blocks_out

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################

    # helper function
    @classmethod
    def get_block_sizes(cls, row_reps, col_reps, signature):
        """
        Compute shapes of blocks authorized with row_reps and col_reps and their
        associated irreps

        Parameters
        ----------
        row_reps : tuple of representations
            Row representations
        col_reps : tuple of representations
            Column representations
        signature : 1D bool array
            Signature on each leg.

        Returns
        -------
        block_irreps : integer array
            Irreducible representations for each block
        block_shapes : list of 2-tuple
            Shape of each block
        """
        row_tot = cls.combine_representations(row_reps, signature[: len(row_reps)])
        col_tot = cls.combine_representations(col_reps, signature[len(row_reps) :])
        i1 = 0
        i2 = 0
        block_irreps = []
        block_shapes = []
        while i1 < row_tot.shape[1] and i2 < col_tot.shape[1]:
            if row_tot[1, i1] == col_tot[1, i2]:  # if irreps are the same
                sh = (row_tot[0, i1], col_tot[0, i2])  # degeneracies
                block_irreps.append(row_tot[1, i1])
                block_shapes.append(sh)
                i1 += 1
                i2 += 1
            elif row_tot[1, i1] < col_tot[1, i2]:
                i1 += 1
            else:
                i2 += 1
        return np.array(block_irreps), block_shapes

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

        # define a generic function that precompute elementary blocks
        # as well as their block_irreps, extern and inner row and col degen?

        block_irreps, block_shapes_in = cls.get_block_sizes(
            row_reps, col_reps, signature
        )
        nblocks = block_irreps.size
        shifts = []
        elementary_block_per_axis = np.empty((ndim,), dtype=int)
        for ax, rep in enumerate(in_reps):
            s = [0]
            elementary_block_per_axis[ax] = rep.shape[1]
            for i in range(rep.shape[1]):
                s.append(cls.representation_dimension(rep[:, : i + 1]))
            shifts.append(s)
        degen_irrep_perm = tuple(range(1, 2 * ndim, 2)) + tuple(range(0, 2 * ndim, 2))

        # determine elementary block number
        n_ele_row = elementary_block_per_axis[:nrr].prod()
        n_ele_col = elementary_block_per_axis[nrr:].prod()
        n_ele = n_ele_row * n_ele_col

        # decompose ele counter as a multi index to recover index of irrep in each axis
        ele_reps_indices = np.array(
            np.unravel_index(np.arange(n_ele), elementary_block_per_axis)
        ).T.reshape(n_ele_row, n_ele_col, ndim)

        blocks = tuple(np.empty(sh) for sh in block_shapes_in)
        block_shifts_row = np.zeros((nblocks,), dtype=int)
        ele_sign = np.zeros((ndim - nrr + 1,), dtype=bool)
        ele_sign[1:] = signature[nrr:]

        # loop over row elementary blocks
        for ir_ele in range(n_ele_row):
            ele_r_in = []
            mul_ind = ele_reps_indices[ir_ele, 0]
            ele_degen_dimensions = np.empty((ndim, 2), dtype=int)

            for i in range(nrr):
                rep = in_reps[i][:, mul_ind[i]]
                ele_degen_dimensions[i, 0] = rep[0]
                ele_degen_dimensions[i, 1] = cls.irrep_dimension(rep[1])
                ele_r_in.append(np.array([[1], [rep[1]]]))

            edir = ele_degen_dimensions[:nrr, 0].prod()
            rep_row_in = cls.combine_representations(ele_r_in, signature[:nrr])

            # reset columns shifts
            block_shifts_col = np.zeros((nblocks,), dtype=int)

            # loop over column elementary blocks
            for ic_ele in range(n_ele_col):
                mul_ind = ele_reps_indices[ir_ele, ic_ele]

                # find irreps appearing in this elementary block
                ele_c_in = []
                for i in range(nrr, ndim):
                    rep = in_reps[i][:, mul_ind[i]]
                    ele_degen_dimensions[i, 0] = rep[0]
                    ele_degen_dimensions[i, 1] = cls.irrep_dimension(rep[1])
                    ele_c_in.append(np.array([[1], [rep[1]]]))

                block_irreps_ele, block_shapes_in_ele = cls.get_block_sizes(
                    [rep_row_in], ele_c_in, ele_sign
                )

                if block_irreps_ele.size:  # if there is a singlet
                    edic = ele_degen_dimensions[nrr:, 0].prod()
                    edi = edir * edic

                    # construct CG projector on elementary sector
                    p_in = cls.construct_matrix_projector(ele_r_in, ele_c_in, signature)

                    # construct elementary block sector in dense tensor
                    slices = tuple(
                        slice(shifts[i][mul_ind[i]], shifts[i][mul_ind[i] + 1], 1)
                        for i in range(ndim)
                    )
                    arr_ele = arr[slices]

                    # split degen and structural
                    sh = ele_degen_dimensions.ravel()
                    data = arr_ele.reshape(sh).transpose(degen_irrep_perm)
                    data = data.reshape(ele_degen_dimensions[:, 1].prod(), edi)

                    # slice to construct matrix blocks
                    ishift = 0
                    for ibi, irr in enumerate(block_irreps_ele):
                        idirb, idicb = block_shapes_in_ele[ibi]
                        idib = idirb * idicb
                        norm = np.sqrt(cls.irrep_dimension(irr))
                        block_ele_proj = p_in[:, ishift : ishift + idib].T / norm
                        ishift += idib
                        data_block = block_ele_proj @ data

                        # data_block still has a non-trivial structure due to inner
                        # degeneracies. Its shape is
                        # (int_degen_row, int_degen_col, ext_degen_row, ext_degen_col)
                        # we need to permute axes to reshape as
                        # (ext_degen_row * int_degen_row, ext_degen_col * int_degen_col)
                        sh = (edir * idirb, edic * idicb)
                        data_block = data_block.reshape(idirb, idicb, edir, edic)
                        data_block = data_block.swapaxes(1, 2).reshape(sh)

                        bi = block_irreps.searchsorted(irr)
                        rs = slice(block_shifts_row[bi], block_shifts_row[bi] + sh[0])
                        cs = slice(block_shifts_col[bi], block_shifts_col[bi] + sh[1])
                        blocks[bi][rs, cs] = data_block
                        block_shifts_col[bi] += sh[1]
                    assert ishift == p_in.shape[1]

            # set shifts for rows
            for bi in range(nblocks):
                assert block_shifts_col[bi] in [0, blocks[bi].shape[1]]
                bir = rep_row_in[1].searchsorted(block_irreps[bi])
                if bir < rep_row_in.shape[1] and rep_row_in[1, bir] == block_irreps[bi]:
                    block_shifts_row[bi] += edir * rep_row_in[0, bir]

        st = cls(row_reps, col_reps, blocks, block_irreps, signature)
        assert abs(st.norm() - lg.norm(arr)) <= 1e-13 * lg.norm(arr)
        return st

    def toarray(self, as_matrix=False):
        out_reps = self._row_reps + self._col_reps

        shifts = []
        elementary_block_per_axis = np.empty((self._ndim,), dtype=int)
        for ax, rep in enumerate(out_reps):
            s = [0]
            elementary_block_per_axis[ax] = rep.shape[1]
            for i in range(rep.shape[1]):
                s.append(self.representation_dimension(rep[:, : i + 1]))
            shifts.append(s)

        # determine elementary block number
        n_ele_row = elementary_block_per_axis[: self._nrr].prod()
        n_ele_col = elementary_block_per_axis[self._nrr :].prod()
        n_ele = n_ele_row * n_ele_col

        # decompose ele counter as a multi index to recover index of irrep in each axis
        ele_reps_indices = np.array(
            np.unravel_index(np.arange(n_ele), elementary_block_per_axis)
        ).T.reshape(n_ele_row, n_ele_col, self._ndim)

        ele_degen_dimensions = np.empty((self._ndim, 2), dtype=int)
        block_shifts_row = np.zeros((self._nblocks,), dtype=int)
        degen_irrep_perm = tuple(range(1, 2 * self._ndim, 2)) + tuple(
            range(0, 2 * self._ndim, 2)
        )
        reverse_perm = np.argsort(degen_irrep_perm)

        ele_sign = self._signature[self._nrr - 1 :].copy()
        ele_sign[0] = False

        arr = np.zeros(self._shape, dtype=self.dtype)

        # loop over row elementary blocks
        for ir_ele in range(n_ele_row):
            ele_r_out = []
            mul_ind = ele_reps_indices[ir_ele, 0]

            for i in range(self._nrr):
                rep = out_reps[i][:, mul_ind[i]]
                ele_degen_dimensions[i, 0] = rep[0]
                ele_degen_dimensions[i, 1] = self.irrep_dimension(rep[1])
                ele_r_out.append(np.array([[1], [rep[1]]]))

            edor = ele_degen_dimensions[: self._nrr, 0].prod()
            rep_row_out = self.combine_representations(
                ele_r_out, self._signature[: self._nrr]
            )

            # reset columns shifts
            block_shifts_col = np.zeros((self._nblocks,), dtype=int)

            # loop over column elementary blocks
            for ic_ele in range(n_ele_col):
                mul_ind = ele_reps_indices[ir_ele, ic_ele]

                # find irreps appearing in this elementary block
                ele_c_out = []
                for i in range(self._nrr, self._ndim):
                    rep = out_reps[i][:, mul_ind[i]]
                    ele_degen_dimensions[i, 0] = rep[0]
                    ele_degen_dimensions[i, 1] = self.irrep_dimension(rep[1])
                    ele_c_out.append(np.array([[1], [rep[1]]]))

                block_irreps_ele, block_shapes_out_ele = self.get_block_sizes(
                    [rep_row_out], ele_c_out, ele_sign
                )

                if block_irreps_ele.size:  # if there is a singlet
                    # construct CG projector on elementary sector
                    p_out = self.construct_matrix_projector(
                        ele_r_out, ele_c_out, self._signature
                    )

                    edoc = ele_degen_dimensions[self._nrr :, 0].prod()
                    edo = edor * edoc
                    ele_dense = np.zeros((p_out.shape[0], edo), dtype=self.dtype)

                    # need to loop over all authorized blocks to get correct oshift
                    oshift = 0
                    for bi, irr in enumerate(block_irreps_ele):
                        idorb, idocb = block_shapes_out_ele[bi]
                        idob = idorb * idocb
                        bi_self = self._block_irreps.searchsorted(irr)
                        if (
                            bi_self < self._nblocks
                            and self._block_irreps[bi_self] == irr
                        ):
                            # split degen and structural
                            rs = slice(
                                block_shifts_row[bi_self],
                                block_shifts_row[bi_self] + edor * idorb,
                                1,
                            )
                            cs = slice(
                                block_shifts_col[bi_self],
                                block_shifts_col[bi_self] + edoc * idocb,
                                1,
                            )
                            block_ele_proj = p_out[:, oshift : oshift + idob]
                            norm = np.sqrt(self.irrep_dimension(irr))
                            block_ele_proj = norm * block_ele_proj

                            data_block = self._blocks[bi_self][rs, cs]
                            data_block = data_block.reshape(idorb, edor, idocb, edoc)
                            data_block = data_block.swapaxes(1, 2).reshape(idob, edo)
                            ele_dense += block_ele_proj @ data_block
                            block_shifts_col[bi_self] += edoc * idocb

                        oshift += idob
                    assert oshift == p_out.shape[1]

                    # construct elementary block sector in dense tensor
                    sh_ele = ele_degen_dimensions.T[::-1].ravel()
                    ele_dense = ele_dense.reshape(sh_ele).transpose(reverse_perm)
                    slices = tuple(
                        slice(shifts[i][mul_ind[i]], shifts[i][mul_ind[i] + 1], 1)
                        for i in range(self._ndim)
                    )
                    arr[slices] = ele_dense.reshape(ele_degen_dimensions.prod(axis=1))

            # set shifts for rows
            for idorb, irr in rep_row_out.T:
                bi_self = self._block_irreps.searchsorted(irr)
                if bi_self < self._nblocks and self._block_irreps[bi_self] == irr:
                    assert block_shifts_col[bi_self] in [
                        0,
                        self._blocks[bi_self].shape[1],
                    ]
                    block_shifts_row[bi_self] += edor * idorb

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

        si = int(2 ** np.arange(self._ndim) @ self._signature)
        key = [self._ndim, self._nrr, si, nrr_out, *axes]
        for r in self._row_reps + self._col_reps:
            key.append(r.shape[1])
            key.extend(r[1:].ravel())
        key = tuple(key)
        try:
            structural_data = self._structural_data_dic[key]
        except KeyError:
            structural_data = self._compute_structural_data(axes, nrr_out)
            self._structural_data_dic[key] = structural_data

        degen_data = self._compute_degen_data(axes, nrr_out, structural_data)
        block_irreps, blocks = self._transpose_data(
            axes, nrr_out, structural_data, degen_data
        )

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
        assert abs(ret.norm() - self.norm()) < 1e-13 * self.norm()
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
