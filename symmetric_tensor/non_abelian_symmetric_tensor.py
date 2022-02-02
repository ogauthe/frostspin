import bisect

import numpy as np
import scipy.linalg as lg
import scipy.sparse as ssp

from .symmetric_tensor import SymmetricTensor


class NonAbelianSymmetricTensor(SymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with non-abelian symmetry.
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    # every method is group-specific

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    _unitary_dic = NotImplemented

    @classmethod
    def construct_matrix_projector(row_reps, col_reps, conjugate_columns=False):
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
        conjugate_columns : bool
            Whether to add projector on singlet on column axes.

        Returns
        -------
        proj : (M, N) sparse matrix
            Projector on singlet, with M the dimension of the full parameter space and
            N the singlet space dimension.
        """
        raise NotImplementedError("Must be defined in derived class")

    @classmethod
    def load_unitaries(cls, savefile):
        root = "_ST_unitary_"
        with np.load(savefile) as fin:
            if fin["_ST_symmetry"] != cls.symmetry:
                raise ValueError("Savefile symmetry do not match SymmetricTensor")
            keys = fin[root + "keys"]
            for k in keys:
                words = k.split(";")
                w0 = int(words[0])
                w1 = tuple(int(w) for w in words[1][1:-1].split(",") if w)
                w2 = tuple(int(w) for w in words[2][1:-1].split(",") if w)
                t = tuple(np.array(w.split(), dtype=int).tobytes() for w in words[3:])
                nk = (w0, w1, w2) + t
                data = fin[root + k + "_data"]
                indices = fin[root + k + "_indices"]
                indptr = fin[root + k + "_indptr"]
                sh = fin[root + k + "_shape"]
                cls._unitary_dic[nk] = ssp.csr_matrix((data, indices, indptr), shape=sh)

    @classmethod
    def save_unitaries(cls, savefile):
        data = {"_ST_symmetry": cls.symmetry}
        keys = []
        root = "_ST_unitary_"
        # cannot use dict key directly as savefile keyword: it has to be a valid zip
        # filename, which decoded bytes are not (some values are not allowed)
        for (k, v) in cls._unitary_dic.items():
            nk = ";".join([str(np.frombuffer(b, dtype=int))[1:-1] for b in k[3:]])
            nk = f"{k[0]};{k[1]};{k[2]};" + nk
            keys.append(nk)
            data[root + nk + "_data"] = v.data
            data[root + nk + "_indices"] = v.indices
            data[root + nk + "_indptr"] = v.indptr
            data[root + nk + "_shape"] = v.shape
        data[root + "keys"] = np.array(keys)
        np.savez_compressed(savefile, **data)

    ####################################################################################
    # Non-abelian shared symmetry implementation
    ####################################################################################
    def construct_unitary(self, new_row_reps, new_col_reps, axes):
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
            Representation of transposed tensor row legs
        new_col_reps: tuple of representations
            Representation of transposed tensor column legs
        axes: tuple of int
            axes permutation
        """
        proj1 = self.construct_matrix_projector(self._row_reps, self._col_reps)
        proj2 = self.construct_matrix_projector(new_row_reps, new_col_reps)

        # so, now we have initial shape projector and output shape projector. We need to
        # transpose rows to contract them. Since there is no bra/ket exchange, this can
        # be done by pure advanced slicing, without calling heavier sparse_transpose.
        nsh = tuple(
            self.representation_dimension(rep) for rep in new_row_reps + new_col_reps
        )
        strides1 = np.array((1,) + self._shape[:0:-1]).cumprod()[::-1]
        strides2 = np.array((1,) + nsh[:0:-1]).cumprod()[::-1]
        perm = (np.arange(proj1.shape[0])[:, None] // strides1 % self._shape)[
            :, axes
        ] @ strides2

        proj2 = proj2[perm].T.tocsr()
        unitary = proj2 @ proj1
        # tests show that construct_matrix_projector output has no numerical zeros
        # however unitary may have more than 70% stored coeff being numerical zeros,
        # with several order of magnitude between them and real non-zeros.
        unitary.data[np.abs(unitary.data) < 1e-14] = 0.0
        unitary.eliminate_zeros()
        unitary = unitary.sorted_indices()  # copy to get clean data array
        return unitary

    @classmethod
    def _blocks_from_raw_data(cls, raw_data, row_rep, col_rep):
        # TODO put sqrt(dim) in "unitary" and jit me
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
        return blocks, block_irreps

    def to_raw_data(self):
        # some blocks may be missing, yet raw_data has to include the corresponding
        # zeros at the accurate position. This is inefficient.
        # TODO: remove this method
        row_rep = self.get_row_representation()
        col_rep = self.get_column_representation()
        shared, indL, indR = np.intersect1d(  # bruteforce numpy > clever python
            row_rep[1],
            col_rep[1],
            assume_unique=True,
            return_indices=True,
        )
        data = np.zeros(row_rep[0, indL] @ col_rep[0, indR])
        k = 0
        # hard to jit, need to call self._blocks[i] which may be heterogenous
        for i, irr in enumerate(shared):
            j = bisect.bisect_left(self._block_irreps, irr)
            if j < self._nblocks and self._block_irreps[j] == irr:
                b = self._blocks[j]
                data[k : k + b.size] = b.ravel() * np.sqrt(self.irrep_dimension(irr))
                k += b.size
            else:  # missing block
                k += row_rep[0, indL[i]] * col_rep[0, indR[i]]
        return data

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, conjugate_columns=True):
        assert arr.shape == tuple(
            cls.representation_dimension(rep) for rep in row_reps + col_reps
        )
        proj = cls.construct_matrix_projector(
            row_reps, col_reps, conjugate_columns=conjugate_columns
        )
        raw_data = proj.T @ arr.ravel()
        if conjugate_columns:
            col_reps = tuple(cls.conjugate_representation(r) for r in col_reps)
        blocks, block_irreps = cls._blocks_from_raw_data(
            raw_data,
            cls.combine_representations(*row_reps),
            cls.combine_representations(*col_reps),
        )
        assert abs(
            (n := lg.norm(arr))
            - np.sqrt(
                sum(
                    cls.irrep_dimension(irr) * lg.norm(b) ** 2
                    for (irr, b) in zip(block_irreps, blocks)
                )
            )
            <= 2e-13 * n  # allows for arr = 0
        ), "norm is not conserved in SymmetricTensor cast"
        return cls(row_reps, col_reps, blocks, block_irreps)

    def _toarray(self):
        proj = self.construct_matrix_projector(
            self._row_reps, self._col_reps, conjugate_columns=True
        )
        arr = proj @ self.to_raw_data()
        return arr.reshape(self.matrix_shape)

    def _permutate(self, row_axes, col_axes):
        nx0 = len(self._row_reps)
        row_reps = []
        for ax in row_axes:
            if ax < nx0:
                row_reps.append(self._row_reps[ax])
            else:
                row_reps.append(self.conjugate_representation(self._col_reps[ax - nx0]))
        col_reps = []
        for ax in col_axes:
            if ax < nx0:
                col_reps.append(self.conjugate_representation(self._row_reps[ax]))
            else:
                col_reps.append(self._col_reps[ax - nx0])

        # if hash is too slow, can be decomposed: at init, set dic corresponding to
        # representations, then permutate only needs hash from row_axes and col_axes
        key = tuple(r.tobytes() for r in self._row_reps + self._col_reps)
        key = (len(self._row_reps), row_axes, col_axes) + key
        try:
            unitary = self._unitary_dic[key]
        except KeyError:
            unitary = self.construct_unitary(row_reps, col_reps, row_axes + col_axes)
            self._unitary_dic[key] = unitary
        # TODO slice unitary to do product blockwise, allowing for missing blocks
        # also include sqrt(dim) in input and output
        raw_data = unitary @ self.to_raw_data()
        blocks, block_irreps = self._blocks_from_raw_data(
            raw_data,
            self.combine_representations(*row_reps),
            self.combine_representations(*col_reps),
        )
        return type(self)(row_reps, col_reps, blocks, block_irreps)

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

    def norm(self):
        n2 = 0.0
        for (irr, b) in zip(self._block_irreps, self._blocks):
            n2 += self.irrep_dimension(irr) * lg.norm(b) ** 2
        return np.sqrt(n2)
