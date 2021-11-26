import numpy as np
import scipy.linalg as lg

from misc_tools.svd_tools import sparse_svd, numba_find_chi_largest


# choices are made to make code light and fast:
# irreps are labelled by integers (non-simple groups need another class using lexsort)
# axes are grouped into either row or colum axes to define block-diagonal matrices
# coefficients are stored as dense blocks corresponding to irreps
# blocks are always sorted according to irreps
# blocks are tuple of contiguous F or C arrays (numba)


class SymmetricTensor:
    """
    Generic base class to deal with symmetric tensors. Defines interface that can be
    implemented as derived classes for any symmetry.

    Tensors are seen as matrices, with legs grouped into two packs, rows and columns.
    Some authorized blocks are allowed to be missing if they are 0. Such cases arises
    from a matmul call when irrep X appears on left rows and right columns but not in
    the middle representation.
    """

    ####################################################################################
    # Symmetry implementation
    # each of those methods must be defined according to group rules to define symmetry.
    ####################################################################################
    @classmethod
    @property
    def symmetry(cls):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def combine_representations(*reps):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def conjugate_representation(rep):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def init_representation(degen, irreps):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def representation_dimension(rep):
        raise NotImplementedError("Must be defined in derived class")

    ####################################################################################
    # Symmetry specific methods with fixed signature
    # These methods must be defined in subclasses to set SymmetricTensor behavior
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, conjugate_columns=True):
        raise NotImplementedError("Must be defined in derived class")

    def _toarray(self):
        # returns rank-2 numpy array, called by public toarray
        raise NotImplementedError("Must be defined in derived class")

    def _permutate(self, row_axes, col_axes):
        # returns SymmetricTensor, called by public permutate after input check and
        # cast to C-array only
        raise NotImplementedError("Must be defined in derived class")

    def group_conjugated(self):
        """
        Return a new tensor with all representations (row, columns and blocks irreps)
        conjugated according to group rules. This may change block order, but not the
        block themselves. Since the tensor is a group singlet, it is unaffected in its
        dense form.
        """
        raise NotImplementedError("Must be defined in derived class")

    def check_blocks_fit_representation(self):
        raise NotImplementedError("Must be defined in derived class")

    def norm(self):
        """
        Tensor Frobenius norm.
        """
        raise NotImplementedError("Must be defined in derived class")

    ####################################################################################
    # Initializer
    ####################################################################################
    def __init__(self, row_reps, col_reps, blocks, block_irreps):
        self._row_reps = tuple(row_reps)
        self._col_reps = tuple(col_reps)
        self._shape = tuple(
            self.representation_dimension(r) for r in self._row_reps + self._col_reps
        )
        self._ndim = len(self._shape)
        self._nblocks = len(blocks)
        if all(b.flags["C"] for b in blocks):
            self._blocks = tuple(blocks)
            self._f_contiguous = False
        elif all(b.flags["F"] for b in blocks):
            self._blocks = tuple(blocks)
            self._f_contiguous = True
        else:
            self._blocks = tuple(np.ascontiguousarray(b) for b in blocks)
            self._f_contiguous = False
        self._block_irreps = np.asarray(block_irreps)
        self._ncoeff = sum(b.size for b in blocks)
        assert self._nblocks > 0
        assert 0 < len(self._row_reps) < self._ndim
        assert self._block_irreps.size == self._nblocks
        assert sorted(set(block_irreps)) == list(block_irreps)
        assert self.check_blocks_fit_representations()

    ####################################################################################
    # getters
    ####################################################################################
    @property
    def nblocks(self):
        return self._nblocks

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

    @property
    def matrix_shape(self):
        nrr = len(self._row_reps)
        return (np.prod(self._shape[:nrr]), np.prod(self._shape[nrr:]))

    @property
    def dtype(self):
        return self._blocks[0].dtype

    @property
    def f_contiguous(self):
        return self._f_contiguous

    @property
    def ncoeff(self):
        return self._ncoeff

    @property
    def blocks(self):
        return self._blocks

    @property
    def block_irreps(self):
        return self._block_irreps

    @property
    def row_reps(self):
        return self._row_reps

    @property
    def col_reps(self):
        return self._col_reps

    ####################################################################################
    # Magic methods
    ####################################################################################
    def __add__(self, other):
        assert self.match_representations(other)
        # need to take into account possibly missing block in self or other
        blocks = []
        block_irreps = []
        i1, i2 = 0, 0
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_irreps[i1] == other._block_irreps[i2]:
                blocks.append(self._blocks[i1] + other._blocks[i2])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < other._block_irreps[i2]:
                blocks.append(self._blocks[i1].copy())  # no data sharing
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
            else:
                blocks.append(other._blocks[i2].copy())  # no data sharing
                block_irreps.append(other._block_irreps[i2])
                i2 += 1

        return type(self)(self._row_reps, self._col_reps, blocks, block_irreps)

    def __sub__(self, other):
        return self + (-other)  # much simplier than dedicated implem for tiny perf loss

    def __mul__(self, x):
        assert np.isscalar(x) or x.size == 1
        blocks = tuple(x * b for b in self._blocks)
        return type(self)(self._row_reps, self._col_reps, blocks, self._block_irreps)

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

    def __neg__(self):
        blocks = tuple(-b for b in self._blocks)
        return type(self)(self._row_reps, self._col_reps, blocks, self._block_irreps)

    def __imul__(self, x):
        for b in self._blocks:
            b *= x
        return self

    def __itruediv__(self, x):
        for b in self._blocks:
            b /= x
        return self

    def __matmul__(self, other):
        """
        Tensor dot operation between two tensors with compatible internal structure.
        Left hand term column axes all are contracted with right hand term row axes.

        Note that some allowed block may be missing in the output tensor, if the
        associated irrep does not appear in the contracted bond.
        """
        assert type(self) == type(other)
        assert (
            self._shape[len(self._row_reps) :] == other._shape[: len(other._row_reps)]
        )
        assert all((r == r2).all() for (r, r2) in zip(self._col_reps, other._row_reps))

        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        while i1 < self._nblocks and i2 < other._nblocks:
            if self._block_irreps[i1] == other._block_irreps[i2]:
                blocks.append(self._blocks[i1] @ other._blocks[i2])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < other._block_irreps[i2]:
                i1 += 1
            else:
                i2 += 1

        return type(self)(self._row_reps, other._col_reps, blocks, block_irreps)

    ####################################################################################
    # misc
    ####################################################################################
    def match_representations(self, st):
        """
        Check if input has same type, same shape and same representations on every legs
        as self. block_irreps and blocks are not checked. Return bool, does not raise.
        """
        return (
            type(self) == type(st)
            and self.shape == st.shape
            and len(self._row_reps) == len(st.row_reps)
            and all((r == r2).all() for (r, r2) in zip(self._row_reps, st._row_reps))
            and all((r == r2).all() for (r, r2) in zip(self._col_reps, st._col_reps))
        )

    def toarray(self, as_matrix=False):
        if self._f_contiguous:  # bug calling numba with f-array unituple
            if as_matrix:
                return self.T._toarray().T
            arr = self.T.toarray()
            k = len(self._col_reps)
            return arr.transpose(*range(k, self._ndim), *range(k))
        m = self._toarray()
        if as_matrix:
            return m
        return m.reshape(self._shape)

    @classmethod
    def random(cls, row_reps, col_reps, conjugate_columns=True, rng=None):
        # aimed for test, dumb implementation with from_array(zero)
        if rng is None:
            rng = np.random.default_rng()
        z = np.zeros([cls.representation_dimension(rep) for rep in row_reps + col_reps])
        st = cls.from_array(z, row_reps, col_reps, conjugate_columns=conjugate_columns)
        st._blocks = tuple(rng.random(b.shape) for b in st._blocks)
        return st

    def copy(self):
        blocks = tuple(b.copy() for b in self._blocks)
        return type(self)(self._row_reps, self._col_reps, blocks, self._block_irreps)

    def get_row_representation(self):
        return self.combine_representations(*self._row_reps)

    def get_column_representation(self):
        return self.combine_representations(*self._col_reps)

    ####################################################################################
    # transpose and permutate
    ####################################################################################
    @property
    def T(self):
        """
        Matrix transpose operation, swapping rows and columns. Internal structure and
        and leg merging are not affected: each block is just transposed. Block irreps
        are group conjugated, which may change block order.
        """
        # Transpose the matrix representation of the tensor, ie swap rows and columns
        # and transpose diagonal blocks, without any data move or copy. Irreps need to
        # be conjugate since row (bra) and columns (ket) are swapped.
        conj = self.group_conjugated()
        blocks = tuple(b.T for b in conj._blocks)
        return type(self)(conj._col_reps, conj._row_reps, blocks, conj._block_irreps)

    def conjugate(self):
        """
        Complex conjugate operation. Block values are conjugate and all representations
        are group conjugated. Internal structure is not affected, however block order
        may change.
        """
        conj = self.group_conjugated()
        blocks = tuple(b.conj() for b in conj._blocks)
        return type(self)(conj._row_reps, conj._col_reps, blocks, conj._block_irreps)

    @property
    def H(self):
        """
        Hermitian conjugate operation, swapping rows and columns and conjugating blocks.
        block_irreps and block order are not affected.
        """
        # block_irreps are conjugate both in T and conj: no change
        blocks = tuple(b.T.conj() for b in self._blocks)
        return type(self)(self._col_reps, self._row_reps, blocks, self._block_irreps)

    def permutate(self, row_axes, col_axes):  # signature != ndarray.transpose
        """
        Permutate axes, changing tensor structure.
        """
        assert sorted(row_axes + col_axes) == list(range(self._ndim))
        nrr = len(self._row_reps)

        # return early for identity or matrix transpose
        if row_axes == tuple(range(nrr)) and col_axes == tuple(range(nrr, self._ndim)):
            return self
        if row_axes == tuple(range(nrr, self._ndim)) and col_axes == tuple(range(nrr)):
            return self.T

        # only permutate C-array (numba bug with tuple of F-array)
        if self._f_contiguous:
            row_axes_T = tuple((ax - nrr) % self._ndim for ax in row_axes)
            col_axes_T = tuple((ax - nrr) % self._ndim for ax in col_axes)
            return self.T._permutate(row_axes_T, col_axes_T)

        return self._permutate(row_axes, col_axes)

    ####################################################################################
    # Linear algebra
    ####################################################################################
    def svd(self):
        u_blocks = [None] * self._nblocks
        s_blocks = [None] * self._nblocks
        v_blocks = [None] * self._nblocks
        for bi, b in enumerate(self._blocks):
            try:
                u, s, v = lg.svd(b, full_matrices=False, check_finite=False)
            except lg.LinAlgError as err:
                print("Error in scipy dense SVD:", err)
                u, s, v = lg.svd(
                    b, full_matrices=False, check_finite=False, driver="gesvd"
                )
            u_blocks[bi] = u
            s_blocks[bi] = s
            v_blocks[bi] = v

        degen = np.array([s.size for s in s_blocks])
        mid_rep = self.init_representation(degen, self._block_irreps)
        U = type(self)(self._row_reps, (mid_rep,), u_blocks, self._block_irreps)
        V = type(self)((mid_rep,), self._col_reps, v_blocks, self._block_irreps)
        return U, s_blocks, V

    def truncated_svd(
        self, cut, max_dense_dim=None, window=0, rcutoff=0.0, degen_ratio=1.0
    ):
        """
        Compute block-wise SVD of self and keep only cut largest singular values. Keep
        only values larger than rcutoff * max(sv).
        """
        if max_dense_dim is None:
            max_dense_dim = 8 * cut

        raw_u = [None] * self._nblocks
        raw_s = [None] * self._nblocks
        raw_v = [None] * self._nblocks
        for bi, b in enumerate(self._blocks):
            if min(b.shape) < max_dense_dim:  # dense svd for small blocks
                try:
                    u, s, v = lg.svd(b, full_matrices=False, check_finite=False)
                except lg.LinAlgError as err:
                    print("Error in scipy dense SVD:", err)
                    u, s, v = lg.svd(
                        b, full_matrices=False, check_finite=False, driver="gesvd"
                    )
            else:
                u, s, v = sparse_svd(b, k=cut + window)
            raw_u[bi] = u
            raw_s[bi] = s
            raw_v[bi] = v

        u_blocks = []
        s_values = []
        v_blocks = []
        block_cuts = numba_find_chi_largest(tuple(raw_s), cut, rcutoff, degen_ratio)
        non_empty = block_cuts.nonzero()[0]
        for bi in non_empty:
            bcut = block_cuts[bi]
            u_blocks.append(np.ascontiguousarray(raw_u[bi][:, :bcut]))
            s_values.append(raw_s[bi][:bcut])
            v_blocks.append(raw_v[bi][:bcut])

        block_irreps = self._block_irreps[non_empty]
        mid_rep = self.init_representation(block_cuts[non_empty], block_irreps)
        U = type(self)(self._row_reps, (mid_rep,), u_blocks, block_irreps)
        V = type(self)((mid_rep,), self._col_reps, v_blocks, block_irreps)
        return U, s_values, V

    def expm(self):
        blocks = tuple(lg.expm(b) for b in self._blocks)
        return type(self)(self._row_reps, self._col_reps, blocks, self._block_irreps)

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
            prefix + "_symmetry": self.symmetry,
            prefix + "_n_row_reps": len(self._row_reps),
            prefix + "_n_col_reps": len(self._col_reps),
            prefix + "_block_irreps": self._block_irreps,
        }
        for ri, r in enumerate(self._row_reps):
            data[f"{prefix}_row_rep_{ri}"] = r
        for ci, c in enumerate(self._col_reps):
            data[f"{prefix}_col_rep_{ci}"] = c
        for bi, b in enumerate(self._blocks):
            data[f"{prefix}_block_{bi}"] = b
        return data

    @classmethod
    def load_from_dic(cls, data, prefix=""):
        if cls.symmetry != data[prefix + "_symmetry"][()]:
            raise ValueError(f"Saved SymmetricTensor does not match type {cls}")
        row_reps = []
        for ri in range(data[prefix + "_n_row_reps"][()]):
            row_reps.append(data[f"{prefix}_row_rep_{ri}"])
        col_reps = []
        for ci in range(data[prefix + "_n_col_reps"][()]):
            col_reps.append(data[f"{prefix}_col_rep_{ci}"])
        block_irreps = data[prefix + "_block_irreps"]
        blocks = []
        for bi in range(block_irreps.size):
            blocks.append(data[f"{prefix}_block_{bi}"])
        return cls(row_reps, col_reps, blocks, block_irreps)

    @classmethod
    def load_from_file(cls, savefile, prefix=""):
        with np.load(savefile) as fin:
            st = cls.load_from_dic(fin, prefix=prefix)
        return st
