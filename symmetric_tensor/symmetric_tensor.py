import numpy as np
import scipy.linalg as lg

from misc_tools.svd_tools import sparse_svd, find_chi_largest
from symmetric_tensor.diagonal_tensor import DiagonalTensor


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

    @classmethod
    @property
    def singlet(cls):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def combine_representations(reps, signature):
        raise NotImplementedError("Must be defined in derived class")

    # conjugate one irrep
    @staticmethod
    def conjugate_irrep(irr):
        raise NotImplementedError("Must be defined in derived class")

    # conjugate a whole representation, may include swapping irreps
    @staticmethod
    def conjugate_representation(rep):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def init_representation(degen, irreps):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def representation_dimension(rep):
        raise NotImplementedError("Must be defined in derived class")

    @staticmethod
    def irrep_dimension(rep):
        raise NotImplementedError("Must be defined in derived class")

    ####################################################################################
    # Symmetry specific methods with fixed signature
    # These methods must be defined in subclasses to set SymmetricTensor behavior
    ####################################################################################
    @classmethod
    def from_array(cls, arr, row_reps, col_reps, signature=None):
        """
        Parameters
        ----------
        arr : ndarray
            Dense array to cast to symmetric blocks.
        row_reps : tuple of arrays
            Representations for the rows.
        row_reps : tuple of arrays
            Representations for the columns.
        signature : 1D boolean array
            Signature of each representation. If None, assumed to be False for rows and
            True for columns.
        """
        raise NotImplementedError("Must be defined in derived class")

    def toarray(self, as_matrix=False):
        raise NotImplementedError("Must be defined in derived class")

    def permutate(self, row_axes, col_axes):  # signature != ndarray.transpose
        """
        Permutate axes, changing tensor structure.

        Parameters
        ----------
        row_axes: tuple of int
            New row axes.
        col_axes: tuple of int
            New column axes.
        """
        raise NotImplementedError("Must be defined in derived class")

    def group_conjugated(self):
        """
        Return a new tensor with all representations (row, columns and blocks irreps)
        conjugated according to group rules. Since the dense tensor is a group singlet,
        it is unaffected in its dense form, however symmetric blocks may change,
        especially in the non-abelian case.
        """
        raise NotImplementedError("Must be defined in derived class")

    def check_blocks_fit_representations(self):
        raise NotImplementedError("Must be defined in derived class")

    def norm(self):
        """
        Tensor Frobenius norm.
        """
        raise NotImplementedError("Must be defined in derived class")

    def toabelian(self):
        """
        Return a SymmetricTensor with largest possible abelian symmetry.
        AsymmetricTensor and AbelianSymmetricTensor are left unchanged.
        """
        raise NotImplementedError("Must be defined in derived class")

    def update_signature(self, sign_update):
        """
        Change signature. This is an in-place operation.

        Parameters
        ----------
        sign_diff: int array, shape (self._ndim)

        Convention:  0: no change
                    +1: change signature, no change in coefficients
                    -1: change in signature with loop

        For abelian symmetries (or asymmetric), this does not alter the blocks and the
        signs in sign_update have no effect. For non-abelian symmetries, this may change
        coefficients because a loop appears in structural tensors.
        """
        raise NotImplementedError("Must be defined in derived class")

    ####################################################################################
    # Initializer
    ####################################################################################
    def __init__(self, row_reps, col_reps, blocks, block_irreps, signature):
        self._row_reps = tuple(row_reps)
        self._col_reps = tuple(col_reps)
        self._shape = tuple(
            self.representation_dimension(r) for r in self._row_reps + self._col_reps
        )
        self._ndim = len(self._shape)
        self._nrr = len(row_reps)
        self._nblocks = len(blocks)
        self._signature = np.ascontiguousarray(signature, dtype=bool)
        self._blocks = tuple(blocks)
        self._block_irreps = np.asarray(block_irreps)
        self._ncoeff = sum(b.size for b in blocks)
        assert self._nblocks > 0
        assert 0 < self._nrr < self._ndim
        assert self._signature.shape == (self.ndim,)
        assert self._block_irreps.size == self._nblocks
        assert sorted(set(block_irreps)) == list(block_irreps)
        assert self.check_blocks_fit_representations()

    def cast(self, symmetry):
        fn = getattr(self, f"to{symmetry}")
        return fn()

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
    def n_row_reps(self):
        return self._nrr

    @property
    def shape(self):
        return self._shape

    @property
    def matrix_shape(self):
        return (np.prod(self._shape[: self._nrr]), np.prod(self._shape[self._nrr :]))

    @property
    def dtype(self):
        return self._blocks[0].dtype

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

    @property
    def signature(self):
        return self._signature

    ####################################################################################
    # Magic methods
    ####################################################################################
    def __repr__(self):
        return f"{self.symmetry} SymmetricTensor with shape {self._shape}"

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

        return type(self)(
            self._row_reps, self._col_reps, blocks, block_irreps, self._signature
        )

    def __sub__(self, other):
        return self + (-other)  # much simplier than dedicated implem for tiny perf loss

    def __mul__(self, other):
        """
        If other is a scalar, this is standard scalar multiplication.

        If other is a DiagonalTensor, this is a blockwise matrix product with a diagonal
        matrix. Symmetries must match.
        """
        if np.issubdtype(type(other), np.number):
            blocks = tuple(other * b for b in self._blocks)
            return type(self)(
                self._row_reps,
                self._col_reps,
                blocks,
                self._block_irreps,
                self._signature,
            )
        if type(other) == DiagonalTensor:  # add diagonal weights on last col leg
            # check DiagonalTensor matches self
            assert self.symmetry == other.symmetry
            assert (self._col_reps[-1] == other.representation).all()
            assert (
                other.block_degen
                == [self.irrep_dimension(irr) for irr in other.block_irreps]
            ).all()
            return self.diagonal_mul(other.diagonal_blocks, other.block_irreps)
        raise TypeError("unsupported operation")

    def __rmul__(self, other):
        if np.issubdtype(type(other), np.number):
            blocks = tuple(other * b for b in self._blocks)
            return type(self)(
                self._row_reps,
                self._col_reps,
                blocks,
                self._block_irreps,
                self._signature,
            )
        if type(other) == DiagonalTensor:  # add diagonal weights on 1st row leg
            assert self.symmetry == other.symmetry
            assert (self._row_reps[0] == other.representation).all()
            assert (
                other.block_degen
                == [self.irrep_dimension(irr) for irr in other.block_irreps]
            ).all()
            return self.diagonal_mul(
                other.diagonal_blocks, other.block_irreps, left=True
            )
        raise TypeError("unsupported operation")

    def diagonal_mul(self, diag_blocks, diag_block_irreps, left=False):
        """
        Matrix product with a diagonal matrix with matching symmetry. If left is True,
        matrix multiplication is from the left.

        Convention: diag_blocks is understood as diagonal weights coming from a SVD in
        terms of representation and signature. Therefore it can only be added to the
        right if self has only one column leg and added to the left if self has only one
        Its signature is assumed to be trivial [False, True]. If it does not match self
        then a signature change is done by conjugating diag_block_irreps.

        Consider calling mul or rmul with a DiagonalTensor object for a more robust
        interface.

        Parameters
        ----------
        diag_blocks : enum of 1D array
            Diagonal block to apply to self.blocks
        diag_block_irreps : int array
            Irreducible representation corresponding to each block
        left : bool
            Whether to multiply from the right (default) or from the left.
        """
        n = len(diag_blocks)
        assert len(diag_block_irreps) == n

        # if signatures do not match, transpose diag_blocks. This requires to conjugate
        # diag_block_irreps and swap blocks to keep them sorted.
        # since signature structure is trivial, block coefficients are not affected.
        if (left and self._signature[0]) or (not left and not self._signature[-1]):
            conj_irreps = np.array([self.conjugate_irrep(r) for r in diag_block_irreps])
            so = conj_irreps.argsort()
            diag_block_irreps = conj_irreps[so]
            diag_blocks = [diag_blocks[i] for i in so]

        if left:
            assert self._nrr == 1
            s = (slice(None, None, None), None)
        else:
            assert self._ndim - self._nrr == 1
            s = None

        i1 = 0
        i2 = 0
        blocks = []
        block_irreps = []
        while i1 < self._nblocks and i2 < n:
            if self._block_irreps[i1] == diag_block_irreps[i2]:
                blocks.append(self._blocks[i1] * diag_blocks[i2][s])
                block_irreps.append(self._block_irreps[i1])
                i1 += 1
                i2 += 1
            elif self._block_irreps[i1] < diag_block_irreps[i2]:
                # the operation is valid but this should not happen for diagonal_irreps
                # coming from a SVD
                print("Warning: missing block in diagonal blocks")
                i1 += 1
            else:
                i2 += 1

        return type(self)(
            self._row_reps, self._col_reps, blocks, block_irreps, self._signature
        )

    def __truediv__(self, x):
        return self * (1.0 / x)

    def __rtruediv__(self, x):
        return self * (1.0 / x)

    def __neg__(self):
        blocks = tuple(-b for b in self._blocks)
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, self._signature
        )

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
        assert self._shape[self._nrr :] == other._shape[: other._nrr]
        assert (self._signature[self._nrr :] ^ other._signature[: other._nrr]).all()
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

        signature = np.hstack(
            (self._signature[: self._nrr], other._signature[other._nrr :])
        )
        return type(self)(
            self._row_reps, other._col_reps, blocks, block_irreps, signature
        )

    ####################################################################################
    # misc
    ####################################################################################
    def match_representations(self, other):
        """
        Check if other has same type, same shape and same representations with same
        signature on every legs as self. block_irreps and blocks are not used.
        Return bool, do not raise.
        """
        if type(self) != type(other):
            return False
        if self._shape != other._shape:
            return False
        if self._nrr != other._nrr:
            return False
        if (self._signature != other._signature).any():
            return False
        for r, r2 in zip(self._row_reps, other._row_reps):
            if r.shape != r2.shape or (r != r2).any():
                return False
        for r, r2 in zip(self._col_reps, other._col_reps):
            if r.shape != r2.shape or (r != r2).any():
                return False
        return True

    @classmethod
    def random(cls, row_reps, col_reps, signature=None, rng=None):
        # aimed for test, dumb implementation with from_array(zero)
        if rng is None:
            rng = np.random.default_rng()
        z = np.zeros([cls.representation_dimension(rep) for rep in row_reps + col_reps])
        st = cls.from_array(z, row_reps, col_reps, signature=signature)
        st._blocks = tuple(rng.random(b.shape) for b in st._blocks)
        return st

    def copy(self):
        # only copy blocks
        # row_reps, col_reps, block_irreps and signature are passed as reference
        blocks = tuple(b.copy() for b in self._blocks)
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, self._signature
        )

    def get_row_representation(self):
        return self.combine_representations(
            self._row_reps, self._signature[: self._nrr]
        )

    def get_column_representation(self):
        return self.combine_representations(
            self._col_reps, ~self._signature[self._nrr :]
        )

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
        raise NotImplementedError("Must be defined in derived class")

    def conjugate(self):
        """
        Complex conjugate operation. Block values are conjugate and all representations
        are group conjugated.
        """
        conj = self.group_conjugated()
        conj._blocks = tuple(b.conj() for b in conj._blocks)
        return conj

    @property
    def H(self):
        """
        Hermitian conjugate operation, swapping rows and columns and conjugating blocks.
        block_irreps and block order are not affected.
        """
        # block_irreps are conjugate both in T and conj: no change
        # conj = self.group_conjugated()
        blocks = tuple(b.T.conj() for b in self._blocks)
        s = ~np.hstack((self.signature[self._nrr :], self._signature[: self._nrr]))
        return type(self)(self._col_reps, self._row_reps, blocks, self._block_irreps, s)

    # TODO raise NotImplementedError and let subclasses deal with it
    def merge_legs(self, i1, i2):
        assert self._signature[i1] == self._signature[i2]
        s = np.array([False, False])
        if i2 < self._nrr:
            assert 0 < i1 + 1 == i2
            r = self.combine_representations(
                (self._row_reps[i1], self._row_reps[i2]), s
            )
            row_reps = self._row_reps[:i1] + (r,) + self._row_reps[i2 + 1 :]
            col_reps = self._col_reps
        else:
            assert self._nrr < i1 + 1 == i2 < self._ndim
            j = i1 - self._nrr
            r = self.combine_representations(
                (self._col_reps[j], self._col_reps[j + 1]), s
            )
            col_reps = self._col_reps[:j] + (r,) + self._col_reps[j + 2 :]
            row_reps = self._row_reps
        signature = np.hstack((self._signature[:i1], self._signature[i2:]))
        return type(self)(
            row_reps, col_reps, self._blocks, self._block_irreps, signature
        )

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
                print("Warning: gesdd returned an error, try gesvd. Error:", err)
                u, s, v = lg.svd(
                    b, full_matrices=False, check_finite=False, lapack_driver="gesvd"
                )
            u_blocks[bi] = u
            s_blocks[bi] = s
            v_blocks[bi] = v

        degen = np.array([s.size for s in s_blocks])
        mid_rep = self.init_representation(degen, self._block_irreps)
        usign = np.hstack((self._signature[: self._nrr], np.array([True])))
        U = type(self)(self._row_reps, (mid_rep,), u_blocks, self._block_irreps, usign)
        degens = [self.irrep_dimension(irr) for irr in self._block_irreps]
        s = DiagonalTensor(s_blocks, mid_rep, self._block_irreps, degens, self.symmetry)
        vsign = np.hstack((np.array([False]), self._signature[self._nrr :]))
        V = type(self)((mid_rep,), self._col_reps, v_blocks, self._block_irreps, vsign)
        return U, s, V

    def truncated_svd(
        self, cut, max_dense_dim=None, window=0, rcutoff=0.0, degen_ratio=1.0
    ):
        """
        Compute block-wise SVD of self and keep only cut largest singular values. Keep
        only values larger than rcutoff * max(sv).

        The truncation tries to preserve multiplets. If all computed singular values are
        kept in a given block (but less than the block size), this may not be possible
        as some values that will be kept in another block were not computed here. This
        may break a non-abelian symmetry, and also produces a suboptimal truncation.
        A warning is displayed whenever this happens.
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
                    print("Warning: gesdd returned an error, try gesvd. Error:", err)
                    u, s, v = lg.svd(
                        b,
                        full_matrices=False,
                        check_finite=False,
                        lapack_driver="gesvd",
                    )
            else:
                u, s, v = sparse_svd(b, k=cut + window)
            raw_u[bi] = u
            raw_s[bi] = s
            raw_v[bi] = v

        u_blocks = []
        s_blocks = []
        v_blocks = []
        dims = np.array([self.irrep_dimension(r) for r in self._block_irreps])
        block_cuts = find_chi_largest(raw_s, cut, dims, rcutoff, degen_ratio)
        non_empty = block_cuts.nonzero()[0]
        warn = 0
        for bi in non_empty:
            bcut = block_cuts[bi]
            warn += bcut == raw_s[bi].size and bcut < min(self._blocks[bi].shape)
            u_blocks.append(np.ascontiguousarray(raw_u[bi][:, :bcut]))
            s_blocks.append(raw_s[bi][:bcut])
            v_blocks.append(raw_v[bi][:bcut])

        if warn:
            print(f"*** WARNING *** kept all computed singular values in {warn} blocks")
        block_irreps = self._block_irreps[non_empty]
        mid_rep = self.init_representation(block_cuts[non_empty], block_irreps)
        usign = np.hstack((self._signature[: self._nrr], np.array([True])))
        U = type(self)(self._row_reps, (mid_rep,), u_blocks, block_irreps, usign)
        degens = [self.irrep_dimension(irr) for irr in block_irreps]
        s = DiagonalTensor(s_blocks, mid_rep, block_irreps, degens, self.symmetry)
        vsign = np.hstack((np.array([False]), self._signature[self._nrr :]))
        V = type(self)((mid_rep,), self._col_reps, v_blocks, block_irreps, vsign)
        return U, s, V

    def expm(self):
        blocks = tuple(lg.expm(b) for b in self._blocks)
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, self._signature
        )

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
            prefix + "_n_row_reps": self._nrr,
            prefix + "_n_col_reps": self._ndim - self._nrr,
            prefix + "_block_irreps": self._block_irreps,
            prefix + "_signature": self._signature,
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
        signature = data[prefix + "_signature"]
        blocks = []
        for bi in range(block_irreps.size):
            blocks.append(data[f"{prefix}_block_{bi}"])
        return cls(row_reps, col_reps, blocks, block_irreps, signature)

    @classmethod
    def load_from_file(cls, savefile, prefix=""):
        with np.load(savefile) as fin:
            st = cls.load_from_dic(fin, prefix=prefix)
        return st
