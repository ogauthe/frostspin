import numpy as np
import scipy.linalg as lg


class DiagonalTensor:
    """
    Diagonal tensor as produced by SVD
    """

    def __init__(
        self, diag_blocks, representation, block_irreps, block_degen, symmetry
    ):
        # store symmetry as a string instead of storing SymmetricTensor type
        # requires to store block_degen (which could be recovered from SymmetricTensor
        # and representation), but makes I/O simpler and avoids cross import
        self._diagonal_blocks = tuple(np.asarray(db) for db in diag_blocks)
        self._representation = representation
        self._block_irreps = np.asarray(block_irreps)
        self._block_degen = np.asarray(block_degen)
        self._symmetry = symmetry
        self._nblocks = len(diag_blocks)
        assert self._block_degen.shape == (self._nblocks,)
        assert self._block_irreps.shape[0] == self._nblocks
        assert all(db.ndim == 1 for db in self._diagonal_blocks)

    ####################################################################################
    # getters
    ####################################################################################
    @property
    def diagonal_blocks(self):
        return self._diagonal_blocks

    @property
    def block_degen(self):
        return self._block_degen

    @property
    def block_irreps(self):
        return self._block_irreps

    @property
    def nblocks(self):
        return self._nblocks

    @property
    def ndim(self):
        return 1

    @property
    def representation(self):
        return self._representation

    @property
    def signature(self):
        return np.array([False, True])

    @property
    def shape(self):
        return (
            sum(
                self._diagonal_blocks[i].size * self._block_degen[i]
                for i in range(self._nblocks)
            ),
        )

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def dtype(self):
        return self._diagonal_blocks[0].dtype

    ####################################################################################
    # Magic methods
    ####################################################################################
    def __repr__(self):
        return f"DiagonalTensor with {self.nblocks} blocks and {self.symmetry} symmetry"

    def __mul__(self, x):
        if np.issubdtype(type(x), np.number):
            blocks = []
            for db in self._diagonal_blocks:
                blocks.append(x * db)
            return type(self)(
                blocks,
                self._representation,
                self._block_irreps,
                self._block_degen,
                self._symmetry,
            )
        return NotImplemented  # call x.__rmul__(self)

    def __imul__(self, x):
        assert np.issubdtype(type(x), np.number)
        for db in self._diagonal_blocks:
            db *= x
        return self

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __itruediv__(self, x):
        self *= 1.0 / x
        return self

    def __pow__(self, x):
        assert np.issubdtype(type(x), np.number)
        blocks = []
        for db in self._diagonal_blocks:
            blocks.append(db**x)
        return type(self)(
            blocks,
            self._representation,
            self._block_irreps,
            self._block_degen,
            self._symmetry,
        )

    ####################################################################################
    # misc
    ####################################################################################
    def sum(self):
        s = 0.0
        for i in range(self._nblocks):
            s += self._block_degen[i] * self._diagonal_blocks[i].sum()
        return s

    def norm(self):
        n2 = 0.0
        for i in range(self._nblocks):
            n2 += self._block_degen[i] * lg.norm(self._diagonal_blocks[i]) ** 2
        return np.sqrt(n2)

    def toarray(self, sort=False):
        """
        Parameters
        ----------
        sort : bool
            Whether to sort weights. Apply SVD weight convention: weights are sorted in
            non-increasing order.

        Returns
        -------
        arr : ndarray
            Weights, grouped by irrep or sorted depending on sort.
        """
        arr = np.empty(self.shape, dtype=self.dtype)
        k = 0
        for i in range(self._nblocks):
            dim = self._diagonal_blocks[i].size
            deg = self._block_degen[i]
            for j in range(deg):
                arr[k : k + dim] = self._diagonal_blocks[i]
                k += dim
        assert k == arr.size
        if sort:
            arr = np.sort(arr)[::-1]  # SVD convention: sort from largest to smallest
        return arr

    ####################################################################################
    # I/O
    ####################################################################################
    def save_to_file(self, savefile):
        """
        Save SymmetricTensor into savefile with npz format.
        """
        data = self.get_data_dic()
        np.savez_compressed(savefile, **data)

    def get_data_dic(self, prefix=""):
        """
        Construct data dictionary containing all information to store the
        SymmetricTensor into an external file.
        """
        # allows to save several SymmetricTensors in one file by using different
        # prefixes.
        data = {
            prefix + "_block_irreps": self._block_irreps,
            prefix + "_block_degen": self._block_degen,
            prefix + "_representation": self._representation,
            prefix + "_symmetry": self.symmetry,
        }
        for bi, db in enumerate(self._diagonal_blocks):
            data[f"{prefix}_diagonal_block_{bi}"] = db
        return data

    @classmethod
    def load_from_dic(cls, data, prefix=""):
        symmetry = data[prefix + "_symmetry"]
        block_irreps = data[prefix + "_block_irreps"]
        block_degen = data[prefix + "_block_degen"]
        representation = data[prefix + "_representation"]
        diag_blocks = []
        for bi in range(block_irreps.size):
            diag_blocks.append(data[f"{prefix}_diagonal_block_{bi}"])
        return cls(diag_blocks, representation, block_irreps, block_degen, symmetry)

    @classmethod
    def load_from_file(cls, savefile, prefix=""):
        with np.load(savefile) as fin:
            st = cls.load_from_dic(fin, prefix=prefix)
        return st
