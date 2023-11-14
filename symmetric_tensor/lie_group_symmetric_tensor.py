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
        # may be optimized by computing separetly cg_tree row and cg_tree col
        # without calling construct_matrix_projector
        in_reps = self._row_reps + self._col_reps

        elementary_block_per_axis = np.array([r.shape[1] for r in in_reps])
        n_ele = elementary_block_per_axis.prod()

        block_irreps_in, _ = self.get_block_sizes(
            self._row_reps, self._col_reps, self._signature
        )
        nblocks_in = len(block_irreps_in)

        # data on elementary block: lists [i_ele]
        ele_reps_indices = np.array(
            np.unravel_index(np.arange(n_ele), elementary_block_per_axis)
        ).T
        contribute_ele = []

        # data on block irreps: array (n_ele, nblocks_in)
        isometry_in_blocks = []
        idirb = []
        idicb = []

        # data on block_irreps out: array (n_ele, nblocks_out)
        idorb = []
        idocb = []

        if signature_update is None:
            perm = tuple([*axes, self._ndim])
            out_signature = self._signature[axes]
            out_reps = tuple(in_reps[i] for i in axes)
            block_irreps_out, _ = self.get_block_sizes(
                out_reps[:nrr_out], out_reps[nrr_out:], out_signature
            )
            nblocks_out = len(block_irreps_out)

            def compute_elementary_unitary(ele_r_in, ele_c_in, ele_r_out, ele_c_out):
                return self._compute_elementary_transpose(
                    ele_r_in, ele_c_in, ele_r_out, ele_c_out, perm, out_signature
                )

        else:
            block_irreps_out = block_irreps_in
            nblocks_out = nblocks_in
            out_signature = self._signature ^ signature_update.astype(bool)

            def compute_elementary_unitary(ele_r_in, ele_c_in, ele_r_out, ele_c_out):
                return self._compute_elementary_conjugate(
                    ele_r_in, ele_c_in, signature_update
                )

        for i_ele in range(n_ele):
            mul_ind = ele_reps_indices[i_ele]

            # find irreps appearing in this elementary block
            in_ele_irreps = np.array(
                [in_reps[i][1, mul_ind[i]] for i in range(self._ndim)]
            )
            ele_in = tuple(np.array([[1], [irr]]) for irr in in_ele_irreps)
            block_irreps_in_ele, block_shapes_in_ele = self.get_block_sizes(
                ele_in[: self._nrr], ele_in[self._nrr :], self._signature
            )
            if block_irreps_in_ele.size:  # if elementary block is allowed
                # compute out only if block contributes
                ele_out = tuple(ele_in[i] for i in axes)
                block_irreps_out_ele, block_shapes_out_ele = self.get_block_sizes(
                    ele_out[:nrr_out], ele_out[nrr_out:], out_signature
                )

                isometry_ele = compute_elementary_unitary(
                    ele_in[: self._nrr],
                    ele_in[self._nrr :],
                    ele_out[:nrr_out],
                    ele_out[nrr_out:],
                )

                # right now isometry_ele is a unitary matrix
                # it is safe to delete smalle values that are numerical zeros
                # typical density around 0.5: keep it as a dense array
                isometry_ele[np.abs(isometry_ele) < 1e-14] = 0
                assert lg.norm(
                    isometry_ele @ isometry_ele.T.conj() - np.eye(isometry_ele.shape[0])
                    < 1e-12
                )

                # We obtained the elementary unitary corresponding to one elementary
                # block basis change. It is probably dense, altough the format is
                # csr_matrix
                # Now we need to apply it to the data.
                # the difficult part is to find the indices where it appears and
                # broadcast it over degeneracies.
                # First we need to slice by block_irreps for both in and out
                # once done, we can easily apply sqrt(irr_in/irr_out) to get an isometry

                idorb_ele = np.zeros((nblocks_out), dtype=int)
                idocb_ele = np.zeros((nblocks_out), dtype=int)
                idirb_ele = np.zeros((nblocks_in), dtype=int)
                idicb_ele = np.zeros((nblocks_in), dtype=int)

                # do not slice columns, but add sqrt(irr_dim) factor
                shift_bo = 0
                for ibo, irr in enumerate(block_irreps_out_ele):
                    ibo_full = block_irreps_out.searchsorted(irr)
                    idorb_ele[ibo_full] = block_shapes_out_ele[ibo][0]
                    idocb_ele[ibo_full] = block_shapes_out_ele[ibo][1]
                    idob = block_shapes_out_ele[ibo][0] * block_shapes_out_ele[ibo][1]
                    norm = np.sqrt(self.irrep_dimension(irr))
                    isometry_ele[shift_bo : shift_bo + idob] /= norm
                    shift_bo += idob
                assert shift_bo == isometry_ele.shape[0]

                filler = np.zeros((0, 0))
                isometry_blocks_ele = [filler] * nblocks_in

                shift_bi = 0
                for ibi, irr in enumerate(block_irreps_in_ele):
                    ibi_full = block_irreps_in.searchsorted(irr)
                    idirb_ele[ibi_full] = block_shapes_in_ele[ibi][0]
                    idicb_ele[ibi_full] = block_shapes_in_ele[ibi][1]
                    idib = block_shapes_in_ele[ibi][0] * block_shapes_in_ele[ibi][1]
                    norm = np.sqrt(self.irrep_dimension(irr))
                    block = norm * isometry_ele[:, shift_bi : shift_bi + idib]
                    shift_bi += idib
                    isometry_blocks_ele[ibi_full] = block
                assert shift_bi == isometry_ele.shape[0]

                contribute_ele.append(i_ele)
                isometry_in_blocks.append(isometry_blocks_ele)
                idirb.append(idirb_ele)
                idicb.append(idicb_ele)
                idorb.append(idorb_ele)
                idocb.append(idocb_ele)

        contribute_ele = np.array(contribute_ele)
        idirb = np.array(idirb)
        idicb = np.array(idicb)
        idorb = np.array(idorb)
        idocb = np.array(idocb)

        structual_data = contribute_ele, idirb, idicb, idorb, idocb, isometry_in_blocks
        return structual_data

    def _compute_degen_data(self, axes, nrr_out, structural_data):
        """
        Compute indices and slices, depending on both internal external degeneracies.
        Symmetry-agnostic, but depends on self degeneracies. Data is not accessed
        """
        # should be possible to compile it
        contribute_ele, idirb, idicb, idorb, idocb, _ = structural_data
        nblocks_in = idirb.shape[1]
        nblocks_out = idorb.shape[1]
        in_reps = self._row_reps + self._col_reps
        elementary_block_per_axis = np.array([r.shape[1] for r in in_reps])

        n_ele_contributing = contribute_ele.size
        ele_reps_indices = np.array(
            np.unravel_index(contribute_ele, elementary_block_per_axis)
        )

        external_degen_in = np.empty((self._ndim, n_ele_contributing), dtype=int)
        for j in range(self._ndim):
            external_degen_in[j] = in_reps[j][0, ele_reps_indices[j]]

        edir = external_degen_in[: self._nrr].prod(axis=0)
        edic = external_degen_in[self._nrr :].prod(axis=0)
        edor = external_degen_in[axes[:nrr_out]].prod(axis=0)
        edoc = external_degen_in[axes[nrr_out:]].prod(axis=0)

        slices_in = np.zeros((4, n_ele_contributing, nblocks_in), dtype=int)
        _, new_row = np.unique(ele_reps_indices[: self._nrr], axis=1, return_index=True)
        new_row = [*new_row, n_ele_contributing]
        dirb = idirb * edir[:, None]
        dicb = idicb * edic[:, None]
        k = 0
        rs = np.zeros((nblocks_in,), dtype=int)
        for nr1, nr2 in zip(new_row, new_row[1:]):
            nk = k + nr2 - nr1
            cs = dicb[nr1:nr2].cumsum(axis=0)
            slices_in[2, k + 1 : nk] = cs[:-1]
            slices_in[3, k:nk] = cs
            slices_in[0, k:nk] += rs
            rs += dirb[nr1:nr2].max(axis=0)
            slices_in[1, k:nk] += rs
            k = nk

        slices_in = np.ascontiguousarray(slices_in.transpose(1, 2, 0))

        # sorting order for elementary blocks in OUT
        block_perm = np.ravel_multi_index(
            ele_reps_indices[axes], elementary_block_per_axis[axes]
        ).argsort()

        slices_out = np.zeros((4, n_ele_contributing, nblocks_out), dtype=int)
        _, new_row = np.unique(
            ele_reps_indices[axes[:nrr_out, None], block_perm],
            axis=1,
            return_index=True,
        )
        new_row = [*new_row, n_ele_contributing]
        dorb = (idorb * edor[:, None])[block_perm]
        docb = (idocb * edoc[:, None])[block_perm]
        k = 0
        rs = np.zeros((nblocks_out,), dtype=int)
        for nr1, nr2 in zip(new_row, new_row[1:]):
            nk = k + nr2 - nr1
            cs = docb[nr1:nr2].cumsum(axis=0)
            slices_out[2, k + 1 : nk] = cs[:-1]
            slices_out[3, k:nk] = cs
            slices_out[0, k:nk] += rs
            rs += dorb[nr1:nr2].max(axis=0)
            slices_out[1, k:nk] += rs
            k = nk

        reverse_perm = block_perm.argsort()
        slices_out = np.ascontiguousarray(slices_out.transpose(1, 2, 0)[reverse_perm])
        degen_data = (external_degen_in.T.copy(), edor, edoc, slices_in, slices_out)
        return degen_data

    def _transpose_data(self, axes, nrr_out, structural_data, degen_data):
        """
        Move data and construct new data blocks
        """
        # 3 nested loops: elementary blocks, rows, columns
        # change loop order?

        _, idirb, idicb, idorb, idocb, isometry_in_blocks = structural_data
        (external_degen_in, edor, edoc, slices_in, slices_out) = degen_data

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

        for i_ele, external_degen_in_ele in enumerate(external_degen_in):
            # edor = external degeneracy out row
            # idicb = internal degeneracy in column block
            edir_tuple = tuple(external_degen_in_ele[: self._nrr])
            edic_tuple = tuple(external_degen_in_ele[self._nrr :])
            edor_ele = edor[i_ele]
            edoc_ele = edoc[i_ele]
            ed = edor_ele * edoc_ele

            out_data = np.zeros((idirb[i_ele] @ idicb[i_ele], ed))

            for ibi, irr in enumerate(block_irreps_in):
                # need to check if this block_irrep_in appears in elementary_block
                # AND if the block exists in tensor
                idirb_ele = idirb[i_ele, ibi]
                idicb_ele = idicb[i_ele, ibi]
                idib = idirb_ele * idicb_ele
                ib_self = self._block_irreps.searchsorted(irr)
                if (
                    idib > 0
                    and ib_self < self._nblocks
                    and self._block_irreps[ib_self] == irr
                ):
                    iso_iblock_ele = isometry_in_blocks[i_ele][ibi]
                    sri1, sri2, sci1, sci2 = slices_in[i_ele, ibi]

                    # there are two operations: changing basis with elementary AND
                    # swapping axes in external degeneracy part. Transpose tensor BEFORE
                    # applying unitary to do only one transpose, before slicing into
                    # out block_irreps

                    in_data = self._blocks[ib_self][sri1:sri2, sci1:sci2]

                    # initial tensor shape = (
                    # internal degeneracy in row block,
                    # *external degeneracies per in row axes,
                    # internal degeneracy in col block,
                    # *external degeneracies per in col axes)
                    sh = (idirb_ele,) + edir_tuple + (idicb_ele,) + edic_tuple
                    in_data = in_data.reshape(sh)

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
                    out_data += iso_iblock_ele @ in_data

            # transpose out_data only once everything is written
            oshift = 0
            for ibo in range(nblocks_out):
                idorb_ele = idorb[i_ele, ibo]
                idocb_ele = idocb[i_ele, ibo]
                idob = idorb_ele * idocb_ele

                if idob > 0:
                    out_block = out_data[oshift : oshift + idob]
                    sh = (idorb_ele, idocb_ele, edor_ele, edoc_ele)
                    out_block = out_block.reshape(sh).swapaxes(1, 2)
                    sh2 = (edor_ele * idorb_ele, edoc_ele * idocb_ele)
                    out_block = out_block.reshape(sh2)

                    # different IN block irreps may contribute: need +=
                    sro1, sro2, sco1, sco2 = slices_out[i_ele, ibo]
                    assert sh2 == (sro2 - sro1, sco2 - sco1)
                    blocks_out[ibo][sro1:sro2, sco1:sco2] = out_block
                    oshift += idob
            assert oshift == idirb[i_ele] @ idicb[i_ele]

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
