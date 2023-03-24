import numpy as np
import scipy.linalg as lg

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
        assert degen.ndim == 1
        assert irreps.shape == (degen.size,)
        return np.vstack((degen, irreps))

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    # group specific

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
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

    def totrivial(self):
        ar = self.toarray()
        rr = self.shape[: self._nrr]
        cr = self.shape[self._nrr :]
        return AsymmetricTensor.from_array(ar, rr, cr, signature=self.signature)
