import bisect

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
    _isometry_dic = NotImplemented

    @classmethod
    def construct_matrix_projector(row_reps, col_reps, signature):
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
    # Non-abelian shared symmetry implementation
    ####################################################################################
    def construct_isometry(self, new_row_reps, new_col_reps, axes):
        r"""
        Construct isometry corresponding to change of non-abelian tensor tree structure.

                    initial
                    /      \
                   /\      /\
                  /\ \    / /\
                  rows1    columns1
                  |||      |||
                 transposition
                  ||||      ||
                  rows2     columns2
                 \/ / /     \  /
                  \/ /       \/
                   \/        /
                    \       /
                      output

        Parameters
        ----------
        new_row_reps: tuple of representations
            Representation of transposed tensor row legs.
        new_col_reps: tuple of representations
            Representation of transposed tensor column legs.
        axes: tuple of int
            axes permutation.

        Returns
        -------
        block_irreps : 1D integer array
            Irreps associated to each isometry blocks.
        iso_blocks : tuple of csc_array
            Isometries to send a given irrep block to its new values. See notes for
            details.

        Notes
        -----
        block_irreps and iso_blocks have same length. For each index i, iso_blocks[i]
        acts on flattened block with same irrep (some blocks may be missing in self) and
        sends it to a flat 1D array which can be reshaped as blocks for the permutated
        tensor. Coefficients sqrt(irrep dimension) are included in the isometry.

        While csr can be slighlty faster than csc, the matrices here are very
        rectangular. They have as many columns as the block they are acting on, but as
        many rows as the number of tensor coefficients. In terms of memory, csr is
        inefficient for such matrices, while csc is great.
        """
        # there are 2 ways to do a basis change: calling to_raw_data, applying some
        # unitary transformation (typically the product of 2 matrix projectors), then
        # calling blocks_from_raw_data. Sqrt(irrep) are applied back and forth in
        # raw_data transformations. Else one can use precomputed isometries which
        # already include these factors. This is faster and cleaner.

        proj1 = self.construct_matrix_projector(
            self._row_reps, self._col_reps, self._signature
        )
        signature = np.array([self._signature[ax] for ax in axes])
        proj2 = self.construct_matrix_projector(new_row_reps, new_col_reps, signature)

        # so, now we have initial shape projector and output shape projector. We need to
        # transpose rows to contract them. Since there is no bra/ket exchange, this can
        # be done by pure advanced slicing, without calling heavier sparse_transpose.
        reps = new_row_reps + new_col_reps
        nsh = tuple(self.representation_dimension(r) for r in reps)
        strides1 = np.array((1,) + self._shape[:0:-1]).cumprod()[::-1]
        strides2 = np.array((1,) + nsh[:0:-1]).cumprod()[::-1]
        n = proj1.shape[0]
        perm = (np.arange(n)[:, None] // strides1 % self._shape)[:, axes] @ strides2

        proj2 = proj2[perm].T.tocsr()
        unitary = proj2 @ proj1
        # tests show that construct_matrix_projector output has no numerical zeros
        # however unitary may have more than 70% stored coeff being numerical zeros,
        # with several order of magnitude between them and real non-zeros.
        unitary.data[np.abs(unitary.data) < 1e-14] = 0.0
        unitary.eliminate_zeros()
        assert lg.norm(
            (unitary @ unitary.T.conj() - ssp.eye(unitary.shape[0])).data
        ) < 1e-13 * np.sqrt(unitary.shape[0]), "unitary transformation is not unitary"

        # add sqrt(irr) factors on output
        iso = unitary.tocoo()
        nnrr = len(new_row_reps)
        rrep = self.combine_representations(new_row_reps, signature[:nnrr])
        crep = self.combine_representations(new_col_reps, signature[nnrr:])
        k, ir, ic = 0, 0, 0
        while ir < rrep.shape[1] and ic < crep.shape[1]:
            if rrep[1, ir] == crep[1, ic]:
                n = rrep[0, ir] * crep[0, ic]
                x = 1 / np.sqrt(self.irrep_dimension(rrep[1, ir]))
                iso.data[(iso.row >= k) * (iso.row < k + n)] *= x  # workaround bug
                k += n
                ir += 1
                ic += 1
            elif rrep[1, ir] < crep[1, ic]:
                ir += 1
            else:
                ic += 1
        assert k == unitary.shape[0]

        # slice isometry + add sqrt on input to get blocks
        iso = iso.tocsc()
        rrep = self.combine_representations(self._row_reps, self.signature[: self._nrr])
        crep = self.combine_representations(self._col_reps, self.signature[self._nrr :])
        blocks, block_irreps = [], []
        k, ir, ic = 0, 0, 0
        while ir < rrep.shape[1] and ic < crep.shape[1]:
            if rrep[1, ir] == crep[1, ic]:
                n = rrep[0, ir] * crep[0, ic]
                p = iso[:, k : k + n] * np.sqrt(self.irrep_dimension(rrep[1, ir]))
                blocks.append(p)
                block_irreps.append(rrep[1, ir])
                k += n
                ir += 1
                ic += 1
            elif rrep[1, ir] < crep[1, ic]:
                ir += 1
            else:
                ic += 1
        assert k == unitary.shape[0]

        return np.array(block_irreps), tuple(blocks)

    @classmethod
    def _blocks_from_raw_data(cls, raw_data, row_rep, col_rep):
        # raw_data transformations are convenient, especially for cast. They should not
        # be often used, performance is not a priority.
        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        k = 0
        while i1 < row_rep.shape[1] and i2 < col_rep.shape[1]:
            if row_rep[1, i1] == col_rep[1, i2]:
                sh = (row_rep[0, i1], col_rep[0, i2])
                m = raw_data[k : k + sh[0] * sh[1]].reshape(sh)
                k += m.size
                blocks.append(m / np.sqrt(cls.irrep_dimension(row_rep[1, i1])))
                block_irreps.append(row_rep[1, i1])
                i1 += 1
                i2 += 1
            elif row_rep[1, i1] < col_rep[1, i2]:
                i1 += 1
            else:
                i2 += 1
        assert k == raw_data.size
        return blocks, block_irreps

    def _to_raw_data(self):
        # raw_data transformations are convenient, especially for cast. They should not
        # be often used, performance is not a priority.
        row_rep = self.get_row_representation()
        col_rep = self.get_column_representation()
        shared, indL, indR = np.intersect1d(  # bruteforce numpy > clever python
            row_rep[1], col_rep[1], assume_unique=True, return_indices=True
        )
        data = np.zeros(row_rep[0, indL] @ col_rep[0, indR])
        k = 0
        # take care of potentially missing blocks
        for i, irr in enumerate(shared):
            j = bisect.bisect_left(self._block_irreps, irr)
            if j < self._nblocks and self._block_irreps[j] == irr:
                b = self._blocks[j]
                data[k : k + b.size] = b.ravel() * np.sqrt(self.irrep_dimension(irr))
                k += b.size
            else:  # missing block
                k += row_rep[0, indL[i]] * col_rep[0, indR[i]]
        assert k == data.size
        return data

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        assert arr.shape == tuple(
            cls.representation_dimension(rep) for rep in row_reps + col_reps
        )
        nrr = len(row_reps)
        if signature is None:
            signature = np.arange(nrr + len(col_reps)) >= nrr
        else:
            signature = np.ascontiguousarray(signature, dtype=bool)
            assert signature.shape == (arr.ndim,)

        proj = cls.construct_matrix_projector(row_reps, col_reps, signature)
        raw_data = proj.T @ arr.ravel()
        blocks, block_irreps = cls._blocks_from_raw_data(
            raw_data,
            cls.combine_representations(row_reps, signature[:nrr]),
            cls.combine_representations(col_reps, signature[nrr:]),
        )
        st = cls(row_reps, col_reps, blocks, block_irreps, signature)
        assert abs(st.norm() - lg.norm(arr)) <= 1e-13 * lg.norm(
            arr
        ), "norm is not conserved in SymmetricTensor cast"
        return st

    def toarray(self, as_matrix=False):
        proj = self.construct_matrix_projector(
            self._row_reps, self._col_reps, self._signature
        )
        raw = self._to_raw_data()
        arr = proj @ raw
        if as_matrix:
            return arr.reshape(self.matrix_shape)
        return arr.reshape(self._shape)

    @property
    def T(self):
        return self.permutate(
            tuple(range(self._nrr, self._ndim)), tuple(range(self._nrr))
        )

    def permutate(self, row_axes, col_axes):
        assert sorted(row_axes + col_axes) == list(range(self._ndim))

        # return early for identity only, matrix transpose is not trivial
        if row_axes == tuple(range(self._nrr)) and col_axes == tuple(
            range(self._nrr, self._ndim)
        ):
            return self

        axes = row_axes + col_axes
        nrr = len(row_axes)
        signature = []
        reps = []
        for ax in axes:
            signature.append(self._signature[ax])
            if ax < self._nrr:
                reps.append(self._row_reps[ax])
            else:
                reps.append(self._col_reps[ax - self._nrr])
        signature = np.array(signature)

        # retrieve or compute unitary
        si = int(2 ** np.arange(self._ndim) @ self._signature)
        key = [self._ndim, self._nrr, si, len(row_axes), *axes]
        for r in self._row_reps + self._col_reps:
            key.append(r.shape[1])
            key.extend(r.flat)
        key = tuple(key)
        try:
            isometry = self._isometry_dic[key]
        except KeyError:
            isometry = self.construct_isometry(reps[:nrr], reps[nrr:], axes)
            self._isometry_dic[key] = isometry

        # compute new blocks as flat array
        # "isometry" is a tuple (block_irreps, iso_blocks). block_irreps is a 1D integer
        # array containg all block_irreps reachable from row and col representations.
        # iso_blocks is a tuple of csr_array arrays, each matrix is an isometry sending
        # the associated block to its new values in a flat array.
        inds = isometry[0].searchsorted(self._block_irreps)
        assert (isometry[0][inds] == self._block_irreps).all()
        flat = np.zeros((isometry[1][0].shape[0],), dtype=self.dtype)
        for bi in range(self._nblocks):
            flat += isometry[1][inds[bi]] @ self._blocks[bi].ravel()

        # reconstruct blocks
        row_rep = self.combine_representations(reps[:nrr], signature[:nrr])
        col_rep = self.combine_representations(reps[nrr:], signature[nrr:])
        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        k = 0
        while i1 < row_rep.shape[1] and i2 < col_rep.shape[1]:
            if row_rep[1, i1] == col_rep[1, i2]:
                sh = (row_rep[0, i1], col_rep[0, i2])
                m = flat[k : k + sh[0] * sh[1]].reshape(sh)
                blocks.append(m)
                block_irreps.append(row_rep[1, i1])
                k += m.size
                i1 += 1
                i2 += 1
            elif row_rep[1, i1] < col_rep[1, i2]:
                i1 += 1
            else:
                i2 += 1
        assert k == flat.size

        ret = type(self)(reps[:nrr], reps[nrr:], blocks, block_irreps, signature)
        assert abs(ret.norm() - self.norm()) < 1e-13 * self.norm()
        return ret
