import bisect

import numpy as np
import scipy.linalg as lg

from symmetric_tensor.symmetric_tensor import SymmetricTensor


class NonAbelianSymmetricTensor(SymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with non-abelian symmetry.
    """

    _unitary_dic = NotImplemented

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
    def construct_matrix_projector(row_reps, col_reps, conjugate_columns=False):
        return NotImplemented

    @classmethod
    def init_representation(cls, degen, irreps):
        return np.array([degen, irreps], dtype=irreps.dtype)

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

    @classmethod
    def from_raw_data(cls, raw_data, row_reps, col_reps):
        # TODO put sqrt(dim) in "unitary" and jit me
        row_rep = cls.combine_representations(*row_reps)
        col_rep = cls.combine_representations(*col_reps)
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
        block_irreps = np.array(block_irreps)
        return cls(row_reps, col_reps, blocks, block_irreps)

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
        return cls.from_raw_data(raw_data, row_reps, col_reps)

    def toarray(self, as_matrix=False):
        if self._f_contiguous:  # bug calling numba with f-array unituple
            if as_matrix:
                return self.T.toarray(as_matrix=True).T
            arr = self.T.toarray()
            k = len(self._col_reps)
            return arr.transpose(tuple(range(k, self._ndim)) + tuple(range(k)))
        proj = self.construct_matrix_projector(
            self._row_reps, self._col_reps, conjugate_columns=True
        )
        arr = proj @ self.to_raw_data()
        if as_matrix:
            return arr.reshape(self.matrix_shape)
        return arr.reshape(self.shape)

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

    def permutate(self, row_axes, col_axes):
        assert sorted(row_axes + col_axes) == list(range(self._ndim))
        nx0 = len(self._row_reps)
        row_reps = []
        for ax in row_axes:
            if ax < nx0:
                row_reps.append(self._row_reps[ax])
            else:
                row_reps.append(self.conjugate_representation(self._col_reps[ax - nx0]))
        row_reps = tuple(row_reps)
        col_reps = []
        for ax in col_axes:
            if ax < nx0:
                col_reps.append(self.conjugate_representation(self._row_reps[ax]))
            else:
                col_reps.append(self._col_reps[ax - nx0])
        col_reps = tuple(col_reps)

        # if hash is too slow, can be decomposed: at init, set dic corresponding to
        # representations, then permutate only needs hash from row_axes and col_axes
        key = tuple(r.tobytes() for r in self._row_reps + self._col_reps)
        key = key + (len(self._row_reps), row_axes, col_axes)
        try:
            unitary = self._unitary_dic[key]
        except KeyError:
            unitary = self.construct_unitary(row_reps, col_reps, row_axes + col_axes)
            self._unitary_dic[key] = unitary
        # TODO slice unitary to do product blockwise, allowing for missing blocks
        # also include sqrt(dim) in input and output
        raw_data = unitary @ self.to_raw_data()
        return self.from_raw_data(raw_data, row_reps, col_reps)

    def norm(self):
        n2 = 0.0
        for (irr, b) in zip(self._block_irreps, self._blocks):
            n2 += self.irrep_dimension(irr) * lg.norm(b) ** 2
        return np.sqrt(n2)
