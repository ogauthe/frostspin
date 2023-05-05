import numpy as np
import scipy.linalg as lg


class DiagonalTensor:
    """
    Diagonal tensor as produced by SVD
    """

    def __init__(self, diag_blocks, representation, block_irreps, symmetric_type):
        assert len(diag_blocks) == len(block_irreps)
        self._diagonal_blocks = tuple(diag_blocks)
        self._representation = representation
        self._block_irreps = np.asarray(block_irreps)
        self._symmetric_type = symmetric_type
        self._nblocks = len(diag_blocks)
        self._shape = (symmetric_type.representation_dimension(representation),)

    ####################################################################################
    # getters
    ####################################################################################
    @property
    def diagonal_blocks(self):
        return self._diagonal_blocks

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
    def shape(self):
        return self._shape

    @property
    def symmetric_type(self):
        return self._symmetric_type

    @property
    def symmetry(self):
        return self._symmetric_type.symmetry

    @property
    def dtype(self):
        return self._blocks[0].dtype

    ####################################################################################
    # Magic methods
    ####################################################################################
    def __repr__(self):
        return f"DiagonalTensor with {self.nblocks} blocks and {self.symmetry} symmetry"

    def __mul__(self, x):
        assert np.isscalar(x) or x.size == 1
        blocks = []
        for db in self._diagonal_blocks:
            blocks.append(x * db)
        return type(self)(
            blocks, self._representation, self._block_irreps, self._symmetric_type
        )

    def __imul__(self, x):
        for db in self._diagonal_blocks:
            db *= x
        return self

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __itruediv__(self, x):
        for db in self._diagonal_blocks:
            db /= x
        return self

    ####################################################################################
    # misc
    ####################################################################################
    def sum(self):
        s = 0.0
        for i in range(self._nblocks):
            d = self._symmetric_type.irrep_dimension(self._block_irreps[i])
            s += d * self._diagonal_blocks[i].sum()
        return s

    def norm(self):
        n2 = 0.0
        for i in range(self._nblocks):
            d = self._symmetric_type.irrep_dimension(self._block_irreps[i])
            n2 += d * lg.norm(self._diagonal_blocks[i]) ** 2
        return np.sqrt(n2)

    def toarray(self, sort=False):
        arr = np.empty(self._shape, dtype=self.dtype)
        k = 0
        for i in range(self._nblocks):
            dim = self._diagonal_blocks[i].size
            deg = self._symmetric_type.irrep_dimension(self._block_irreps[i])
            for j in range(deg):
                arr[k : k + dim] = self._diagonal_blocks[i]
                k += dim
        assert k == arr.size
        if sort:
            arr = np.sort(arr)
        return arr
