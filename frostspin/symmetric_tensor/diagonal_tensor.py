import numpy as np
import scipy.linalg as lg


class DiagonalTensor:
    """
    Diagonal tensor as produced by SVD or eigenvalue computation.
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
        return (len(self),)

    @property
    def dtype(self):
        return self._diagonal_blocks[0].dtype

    ####################################################################################
    # Magic methods
    ####################################################################################
    def __repr__(self):
        return (
            f"DiagonalTensor with {self.nblocks} blocks and {self.symmetry()} symmetry"
        )

    def __len__(self):
        return sum(
            self._diagonal_blocks[i].size * self._block_degen[i]
            for i in range(self._nblocks)
        )

    def __mul__(self, x):
        if isinstance(x, DiagonalTensor):
            if not (self._representation == x.representation).all():
                raise ValueError("Representations do not match")
            blocks = tuple(
                b1 * b2
                for (b1, b2) in zip(
                    self._diagonal_blocks, x.diagonal_blocks, strict=True
                )
            )
        elif np.issubdtype(type(x), np.number):
            blocks = tuple(x * db for db in self._diagonal_blocks)
        else:
            return NotImplemented  # call x.__rmul__(self)
        return type(self)(
            blocks,
            self._representation,
            self._block_irreps,
            self._block_degen,
            self._symmetry,
        )

    def __rmul__(self, x):
        if np.issubdtype(type(x), np.number):
            return self * x
        msg = f"Unsupported operation for type: {type(x)}"
        raise TypeError(msg)

    def __imul__(self, x):
        if not np.issubdtype(type(x), np.number):
            raise NotImplementedError("Invalid type for *=")
        for db in self._diagonal_blocks:
            db[:] *= x
        return self

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        if np.issubdtype(type(x), np.number):
            blocks = tuple(x / db for db in self._diagonal_blocks)
            return type(self)(
                blocks,
                self._representation,
                self._block_irreps,
                self._block_degen,
                self._symmetry,
            )
        raise NotImplementedError("Invalid type for /")

    def __itruediv__(self, x):
        self *= 1.0 / x
        return self

    def __pow__(self, x):
        if not np.issubdtype(type(x), np.number):
            raise NotImplementedError("Invalid type for **")
        blocks = tuple(db**x for db in self._diagonal_blocks)
        return type(self)(
            blocks,
            self._representation,
            self._block_irreps,
            self._block_degen,
            self._symmetry,
        )

    def __neg__(self):
        blocks = tuple(-db for db in self._diagonal_blocks)
        return type(self)(
            blocks,
            self._representation,
            self._block_irreps,
            self._block_degen,
            self._symmetry,
        )

    def __add__(self, other):
        if not isinstance(other, DiagonalTensor):
            raise NotImplementedError("Invalid type for +")
        if not (self._representation == other.representation).all():
            raise ValueError("Representations do not match")
        blocks = tuple(
            b1 + b2
            for (b1, b2) in zip(
                self._diagonal_blocks, other.diagonal_blocks, strict=True
            )
        )
        return type(self)(
            blocks,
            self._representation,
            self._block_irreps,
            self._block_degen,
            self._symmetry,
        )

    def __sub__(self, other):
        return self + (-other)

    def __pos__(self):
        return self

    ####################################################################################
    # misc
    ####################################################################################
    def copy(self):
        return type(self)(
            tuple(db.copy() for db in self._diagonal_blocks),
            self._representation,
            self._block_irreps,
            self._block_degen,
            self._symmetry,
        )

    def symmetry(self):
        return self._symmetry

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

    def toarray(self, *, sort=False):
        """
        Parameters
        ----------
        sort : bool
            Whether to sort weights. Apply SVD weight convention: weights are sorted in
            non-increasing magnitude order.

        Returns
        -------
        arr : ndarray
            Weights, grouped by irrep or sorted depending on sort.
        """
        arr = np.empty(self.shape, dtype=self.dtype)
        k = 0
        for i in range(self._nblocks):
            deg = self._block_degen[i]
            dim = self._diagonal_blocks[i].size
            arr[k : k + deg * dim].reshape(dim, deg)[:] = self._diagonal_blocks[i][
                :, None
            ]
            k += deg * dim
        assert k == arr.size
        if sort:
            # SVD convention: sort from largest to smallest
            arr = arr[np.abs(arr).argsort()[::-1]]
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

    def get_data_dic(self, *, prefix=""):
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
            prefix + "_symmetry": self._symmetry,
        }
        for bi, db in enumerate(self._diagonal_blocks):
            data[f"{prefix}_diagonal_block_{bi}"] = db
        return data

    @classmethod
    def load_from_dic(cls, data, *, prefix=""):
        symmetry = str(data[prefix + "_symmetry"])
        block_irreps = data[prefix + "_block_irreps"]
        block_degen = data[prefix + "_block_degen"]
        representation = data[prefix + "_representation"]
        diag_blocks = tuple(
            data[f"{prefix}_diagonal_block_{bi}"] for bi in range(block_irreps.size)
        )
        return cls(diag_blocks, representation, block_irreps, block_degen, symmetry)

    @classmethod
    def load_from_file(cls, savefile, *, prefix=""):
        with np.load(savefile) as fin:
            return cls.load_from_dic(fin, prefix=prefix)
