import numpy as np

from .symmetric_tensor import SymmetricTensor
from .asymmetric_tensor import AsymmetricTensor


class NonAbelianSymmetricTensor(SymmetricTensor):
    """
    Efficient storage and manipulation for a tensor with non-abelian symmetry.
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    @staticmethod
    def init_representation(degen, irreps):
        # current format for non-abelian representation:
        # irreps are indexed by 1 integer
        # representations are just 2-row integer array
        # 1st row = degen
        # 2nd row = irrep
        rep = np.empty((2, len(degen)), dtype=int)
        rep[0] = degen
        rep[1] = irreps
        return rep

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    # group specific

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################

    @classmethod
    def get_block_sizes(cls, row_reps, col_reps, signature):
        """
        Compute shapes of blocks authorized with row_reps and col_reps and their
        associated irreps

        Parameters
        ----------
        row_reps : enumerable of int64[2, :]
            Row representations
        col_reps : enumerable of int64[2, :]
            Column representations
        signature : bool[:]
            Signature on each leg.

        Returns
        -------
        block_irreps : int64[:]
            Irreducible representations for each block
        block_shapes : int64[:, 2]
            Shape of each block
        """
        # do not use sorting: bruteforce numpy > clever python
        row_tot = cls.combine_representations(row_reps, signature[: len(row_reps)])
        col_tot = cls.combine_representations(col_reps, ~signature[len(row_reps) :])
        rinds, cinds = (row_tot[1, :, None] == col_tot[1]).nonzero()
        block_irreps = np.empty((rinds.size,), dtype=int)
        block_shapes = np.empty((rinds.size, 2), dtype=np.int64)
        for i in range(rinds.size):
            block_irreps[i] = row_tot[1, rinds[i]]
            block_shapes[i] = row_tot[0, rinds[i]], col_tot[0, cinds[i]]
        return block_irreps, block_shapes

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

    def totrivial(self):
        ar = self.toarray()
        rr = tuple(np.array([d]) for d in self._shape[: self._nrr])
        cr = tuple(np.array([d]) for d in self._shape[self._nrr :])
        return AsymmetricTensor.from_array(ar, rr, cr, signature=self.signature)
