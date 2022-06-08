import bisect

import numpy as np
import scipy.linalg as lg
import scipy.sparse as ssp

from .non_abelian_symmetric_tensor import NonAbelianSymmetricTensor


class LieGroupSymmetricTensor(NonAbelianSymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with non-abelian symmetry defined
    by a Lie group. Axis permutation is done using unitary matrices defined by fusion
    trees of representations.
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
    def load_unitaries(cls, savefile):
        with np.load(savefile) as fin:
            if fin["_ST_symmetry"] != cls.symmetry:
                raise ValueError("Savefile symmetry do not match SymmetricTensor")
            n = fin["_ST_n_unitary"]
            for i in range(n):
                legs = fin[f"_ST_unitary_{i}_legs"]
                k = (legs[0], tuple(legs[2 : legs[1] + 2]), tuple(legs[legs[1] + 2 :]))
                reps = [fin[f"_ST_unitary_{i}_rep_{j}"] for j in range(legs.size - 2)]
                k = k + tuple(r.tobytes() for r in reps)
                data = fin[f"_ST_unitary_{i}_data"]
                indices = fin[f"_ST_unitary_{i}_indices"]
                indptr = fin[f"_ST_unitary_{i}_indptr"]
                shape = fin[f"_ST_unitary_{i}_shape"]
                v = ssp.csr_matrix((data, indices, indptr), shape=shape)
                cls._unitary_dic[k] = v

    @classmethod
    def save_unitaries(cls, savefile):
        data = {"_ST_symmetry": cls.symmetry, "_ST_n_unitary": len(cls._unitary_dic)}
        # cannot use dict key directly as savefile keyword: it has to be a valid zip
        # filename, which decoded bytes are not (some values are not allowed)
        # can use workardound with cast to string, but keys becomes very long, may get
        # trouble if larger than 250 char. Just use dumb count and save reps as arrays.
        for i, (k, v) in enumerate(cls._unitary_dic.items()):
            legs = np.array([k[0], len(k[1]), *k[1], *k[2]])
            reps = [np.frombuffer(b, dtype=int) for b in k[3:]]
            for (j, repj) in enumerate(reps):
                data[f"_ST_unitary_{i}_rep_{j}"] = repj
            data[f"_ST_unitary_{i}_legs"] = legs
            data[f"_ST_unitary_{i}_data"] = v.data
            data[f"_ST_unitary_{i}_indices"] = v.indices
            data[f"_ST_unitary_{i}_indptr"] = v.indptr
            data[f"_ST_unitary_{i}_shape"] = v.shape
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
        # THIS MAY BE WRONG
        # when half-integer and interger spins are mixed, errors may appear
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
        unitary = unitary.sorted_indices()  # copy to get clean data array
        assert lg.norm(
            (unitary @ unitary.T.conj() - ssp.eye(unitary.shape[0])).data
        ) < 1e-14 * np.sqrt(unitary.shape[0]), "unitary transformation is not unitary"
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
            row_rep[1], col_rep[1], assume_unique=True, return_indices=True
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
    # TODO FIX ME THERE MAY BE ERRORS
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
        return cls(row_reps, col_reps, blocks, block_irreps, signature)

    def _toarray(self):
        proj = self.construct_matrix_projector(
            self._row_reps, self._col_reps, self._signature
        )
        arr = proj @ self.to_raw_data()
        return arr.reshape(self.matrix_shape)

    def _permutate(self, row_axes, col_axes):
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

        # if hash is too slow, can be decomposed: at init, set dic corresponding to
        # representations, then permutate only needs hash from row_axes and col_axes
        key = tuple(r.tobytes() for r in self._row_reps + self._col_reps)
        si = int(2 ** np.arange(self._ndim) @ self._signature)
        key = (si, self._nrr, row_axes, col_axes) + key
        try:
            unitary = self._unitary_dic[key]
        except KeyError:
            unitary = self.construct_unitary(reps[:nrr], reps[nrr:], axes)
            self._unitary_dic[key] = unitary
        # TODO slice unitary to do product blockwise, allowing for missing blocks
        # also include sqrt(dim) in input and output
        raw_data = unitary @ self.to_raw_data()
        blocks, block_irreps = self._blocks_from_raw_data(
            raw_data,
            self.combine_representations(reps[:nrr], signature[:nrr]),
            self.combine_representations(reps[nrr:], signature[nrr:]),
        )
        return type(self)(reps[:nrr], reps[nrr:], blocks, block_irreps, signature)
