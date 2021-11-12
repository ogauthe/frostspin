import numpy as np
import scipy.linalg as lg

from symmetric_tensor.symmetric_tensor import SymmetricTensor


class NonAbelianSymmetricTensor(SymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with non-abelian symmetry.
    """

    _unitary_dic = NotImplemented

    @classmethod
    def construct_matrix_projector(row_reps, col_reps, conjugate_columns=True):
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
        row_rep = cls.combine_representations(*row_reps)
        col_rep = cls.combine_representations(*col_reps)
        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        k = 0
        while i1 < row_rep.shape[1] and i2 < col_rep.shape[1]:
            if row_rep[1, i1] == col_rep[1, i2]:
                sh = (row_rep[1, i1], col_rep[1, i2])
                m = raw_data[k : k + sh[0] * sh[1]].reshape(sh)
                k += m.size
                blocks.append(m / np.sqrt(cls.representation_dimension(row_rep[0, i1])))
                block_irreps.append(row_rep[0, i1])
                i1 += 1
                i2 += 1
            elif row_rep[0, i1] < col_rep[0, i1]:
                i1 += 1
            else:
                i2 += 1
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
        proj, ind = self.construct_matrix_projector(
            self._row_reps, self._col_reps, conjugate_columns=True
        )
        m = np.zeros((np.prod(self._shape),), dtype=self.dtype)
        m[ind] = proj @ self.to_raw_data()
        if as_matrix:
            return m
        return m.reshape(self.shape)

    def permutate(self, row_axes, col_axes):
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

        k = (tuple(r.tobytes() for r in row_reps), tuple(r.tobytes() for r in col_reps))
        unitary = self._unitary_dic[k]
        raw_data = unitary @ self.to_raw_data()
        return self.from_raw_data(raw_data, row_reps, col_reps)

    def norm(self):
        n2 = 0.0
        for (irr, b) in zip(self._block_irreps, self._blocks):
            n2 += self.representation_dimension(irr) * lg.norm(b) ** 2
        return np.sqrt(n2)
