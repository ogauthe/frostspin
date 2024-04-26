import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg

from froSTspin.misc_tools.svd_tools import (
    find_chi_largest,
    robust_eigh,
    robust_svd,
    sparse_svd,
)

from .diagonal_tensor import DiagonalTensor

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
    # currently _symmetry is a string. Could be a separate class and all the static
    # methods would become class method returning cls._symmetry.singlet()
    # this may change in the future.
    _symmetry = NotImplemented

    @classmethod
    def symmetry(cls):
        return cls._symmetry

    @staticmethod
    def singlet():
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
    def get_block_sizes(cls, row_reps, col_reps, signature):
        """
        Compute shapes of blocks authorized with row_reps and col_reps and their
        associated irreps

        Parameters
        ----------
        row_reps : tuple of representations
            Row representations
        col_reps : tuple of representations
            Column representations
        signature : 1D bool array
            Signature on each leg.

        Returns
        -------
        block_irreps : 1D integer array
            Irreducible representations for each block
        block_shapes : 2D int array
            Shape of each block
        """
        raise NotImplementedError("Must be defined in derived class")

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

    def toarray(self, *, as_matrix=False):
        """
        Cast SymmetricTensor into a dense array.

        Parameters
        ----------
        as_matrix : bool
            Whether to return tensor with its tensor shape (default) or its matrix shape
        """
        mat = self._tomatrix()
        if as_matrix:
            return mat
        return mat.reshape(self._shape)

    def permute(self, row_axes, col_axes):
        """
        Permutate axes, changing tensor structure.

        Parameters
        ----------
        row_axes: enumerable of int
            New row axes.
        col_axes: enumerable of int
            New column axes.
        """
        # input validation
        row_axes = tuple(row_axes)
        col_axes = tuple(col_axes)
        axes = row_axes + col_axes

        if sorted(axes) != list(range(self._ndim)):
            raise ValueError("Axes do not match tensor")

        # return early for identity or matrix transpose
        if row_axes == tuple(range(self._nrr)) and col_axes == tuple(
            range(self._nrr, self._ndim)
        ):
            return self
        if row_axes == tuple(range(self._nrr, self._ndim)) and col_axes == tuple(
            range(self._nrr)
        ):
            return self.transpose()

        signature = np.empty((self._ndim,), dtype=bool)
        reps = [None] * self._ndim
        for i, ax in enumerate(axes):
            signature[i] = self._signature[ax]
            reps[i] = (
                self._row_reps[ax] if ax < self._nrr else self._col_reps[ax - self._nrr]
            )

        # constructing new blocks, private and symmetry specific
        nrr = len(row_axes)
        blocks, block_irreps = self._permute_data(axes, nrr)

        tp = type(self)(reps[:nrr], reps[nrr:], blocks, block_irreps, signature)
        assert abs(self.norm() - tp.norm()) <= 1e-13 * self.norm(), "norm is different"
        return tp

    def check_blocks_fit_representations(self):
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
        assert 0 < self._nrr < self._ndim
        assert self._signature.shape == (self.ndim,)
        assert self._block_irreps.size == self._nblocks
        assert sorted(set(block_irreps)) == list(block_irreps)
        assert self.check_blocks_fit_representations()

    @classmethod
    def from_diagonal_tensor(cls, dt):
        if type(dt) is not DiagonalTensor:
            raise ValueError(f"Invalid input type: {type(dt)}")
        if cls.symmetry() != dt.symmetry():
            raise ValueError("Symmetries do not match")
        row_reps = (dt.representation,)
        col_reps = (dt.representation,)
        blocks = tuple(np.diag(db) for db in dt.diagonal_blocks)
        block_irreps = dt.block_irreps
        signature = np.array([False, True])
        return cls(row_reps, col_reps, blocks, block_irreps, signature)

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
        try:
            dt = self._blocks[0].dtype
        except IndexError:  # pathological case with nblocks = 0
            dt = np.array([]).dtype  # keep numpy convention
        return dt

    @property
    def ncoeff(self):
        """
        Return number of stored coefficients. May be less than number of
        allowed coefficients if some allowed blocks are missing.
        """
        return sum(b.size for b in self._blocks)

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
        return f"{self.symmetry()} SymmetricTensor with shape {self._shape}"

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
        if isinstance(other, DiagonalTensor):  # add diagonal weights on last col leg
            # check DiagonalTensor matches self
            assert self._symmetry == other.symmetry()
            assert (self._col_reps[-1] == other.representation).all()
            assert (
                other.block_degen
                == [self.irrep_dimension(irr) for irr in other.block_irreps]
            ).all()
            return self._diagonal_mul(other.diagonal_blocks, other.block_irreps)
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
        if isinstance(other, DiagonalTensor):  # add diagonal weights on 1st row leg
            assert self._symmetry == other.symmetry()
            assert (self._row_reps[0] == other.representation).all()
            assert (
                other.block_degen
                == [self.irrep_dimension(irr) for irr in other.block_irreps]
            ).all()
            return self._diagonal_mul(
                other.diagonal_blocks, other.block_irreps, left=True
            )
        raise TypeError("unsupported operation")

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
            b[:] *= x
        return self

    def __itruediv__(self, x):
        for b in self._blocks:
            b[:] /= x
        return self

    def __matmul__(self, other):
        """
        Tensor dot operation between two tensors with compatible internal structure.
        Left hand term column axes all are contracted with right hand term row axes.

        Note that some allowed block may be missing in the output tensor, if the
        associated irrep does not appear in the contracted bond.
        """
        assert type(self) is type(other)
        assert self._shape[self._nrr :] == other._shape[: other._nrr]
        assert (self._signature[self._nrr :] ^ other._signature[: other._nrr]).all()
        assert all(
            (r == r2).all()
            for (r, r2) in zip(self._col_reps, other._row_reps, strict=True)
        )

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

        signature = np.empty((self._nrr + other._ndim - other._nrr,), dtype=bool)
        signature[: self._nrr] = self._signature[: self._nrr]
        signature[self._nrr :] = other._signature[other._nrr :]
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
        if (
            type(self) is not type(other)
            or self._shape != other.shape
            or (self._signature != other.signature).any()
        ):
            return False
        for r, r2 in zip(self._row_reps, other.row_reps, strict=True):
            if r.shape != r2.shape or (r != r2).any():
                return False
        for r, r2 in zip(self._col_reps, other.col_reps, strict=True):
            if r.shape != r2.shape or (r != r2).any():
                return False
        return True

    @classmethod
    def random(cls, row_reps, col_reps, signature=None, *, rng=None):
        """
        Initialize random tensor with given representations and signature.

        Parameters
        ----------
        row_reps : tuple of representations
            Row representations
        col_reps : tuple of representations
            Column representations
        signature : 1D bool array
            Signature for each representation. If None, assumed to be False for rows and
            True for columns.
        rng : numpy random Generator
            Random number generator. If None a new instance is initialized.

        Returns
        -------
        st : SymmetricTensor with cls subclass
            Random SymmetricTensor.
        """
        if signature is None:
            signature = np.zeros((len(row_reps) + len(col_reps),), dtype=bool)
            signature[len(row_reps) :] = True
        if rng is None:
            rng = np.random.default_rng()

        block_irreps, block_shapes = cls.get_block_sizes(row_reps, col_reps, signature)
        blocks = []
        for sh in block_shapes:
            b = rng.random(sh) - 0.5
            blocks.append(b)

        st = cls(row_reps, col_reps, blocks, block_irreps, signature)
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
    # transpose and permute
    ####################################################################################
    def transpose(self):
        """
        Matrix transpose operation, swapping rows and columns. This is a specific case
        of permute that can be optimized.
        """
        blocks, block_irreps = self._transpose_data()
        s = self._signature[np.arange(-self._ndim + self._nrr, self._nrr) % self._ndim]
        return type(self)(self._col_reps, self._row_reps, blocks, block_irreps, s)

    def dual(self):
        """
        Construct dual tensor with reverse signature on all legs. No complex conjugation
        involved. Only the arrow direction is reversed, the irreps inside a
        representation stay the same.
        """
        return self.dagger().conjugate().transpose()

    def conjugate(self):
        """
        Complex conjugate coefficient wise. Representations and signature are not
        affected.
        """
        blocks = tuple(b.conj() for b in self._blocks)
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, self._signature
        )

    def dagger(self):
        """
        Return adjoint matrix: swap rows and columns, apply complex conjugation
        coefficient wise and reverse signature for all leg.
        block_irreps and block order are not affected (transpose maps a block irrep to
        its dual, then signature reversal maps it back to origin)

        This is a costless operation, regardless of symmetry.
        """
        blocks = tuple(b.T.conj() for b in self._blocks)
        s = np.empty((self._ndim,), dtype=bool)
        s[: self._ndim - self._nrr] = ~self.signature[self._nrr :]
        s[self._ndim - self._nrr :] = ~self.signature[: self._nrr]
        return type(self)(self._col_reps, self._row_reps, blocks, self._block_irreps, s)

    def merge_legs(self, i1, i2):
        raise NotImplementedError("Must be defined in derived class")

    ####################################################################################
    # Linear algebra
    ####################################################################################
    def norm(self):
        """
        Tensor Frobenius norm.
        """
        n2 = 0.0
        for irr, b in zip(self._block_irreps, self._blocks, strict=True):
            n2 += self.irrep_dimension(irr) * lg.norm(b) ** 2
        return np.sqrt(n2)

    def full_contract(self, other):
        """
        Contract all legs, equialent to Tr(A @ B) as matrices or tensordot(A, B, ndim)
        as tenosrs. Tensors have to match each other representations and signatures.

        The name of this method is subject to change in the future.
        """
        # TBD rename full_dot, bicontract, trace_contract?
        assert self.match_representations(other.dagger())
        x = 0.0
        shared = (self._block_irreps[:, None] == other.block_irreps).nonzero()
        for i1, i2 in zip(*shared, strict=True):
            bx = np.einsum("ij,ji->", self._blocks[i1], other.blocks[i2])
            x += self.irrep_dimension(self._block_irreps[i1]) * bx
        return x

    def qr(self):
        q_blocks = [None] * self._nblocks
        r_blocks = [None] * self._nblocks
        for bi, b in enumerate(self._blocks):
            q, r = lg.qr(b, mode="economic", check_finite=False)
            q_blocks[bi] = q
            r_blocks[bi] = r

        degen = np.array([r.shape[0] for r in r_blocks])
        mid_rep = self.init_representation(degen, self._block_irreps)
        qsign = np.ones((self._nrr + 1,), dtype=bool)
        qsign[: self._nrr] = self._signature[: self._nrr]
        Q = type(self)(self._row_reps, (mid_rep,), q_blocks, self._block_irreps, qsign)
        rsign = np.zeros((self._ndim - self._nrr + 1,), dtype=bool)
        rsign[1:] = self._signature[self._nrr :]
        R = type(self)((mid_rep,), self._col_reps, r_blocks, self._block_irreps, rsign)
        return Q, R

    def rq(self):
        r_blocks = [None] * self._nblocks
        q_blocks = [None] * self._nblocks
        for bi, b in enumerate(self._blocks):
            r, q = lg.rq(b, mode="economic", check_finite=False)
            r_blocks[bi] = r
            q_blocks[bi] = q

        degen = np.array([r.shape[1] for r in r_blocks])
        mid_rep = self.init_representation(degen, self._block_irreps)
        rsign = np.ones((self._nrr + 1,), dtype=bool)
        rsign[: self._nrr] = self._signature[: self._nrr]
        R = type(self)(self._row_reps, (mid_rep,), r_blocks, self._block_irreps, rsign)
        qsign = np.zeros((self._ndim - self._nrr + 1,), dtype=bool)
        qsign[1:] = self._signature[self._nrr :]
        Q = type(self)((mid_rep,), self._col_reps, q_blocks, self._block_irreps, qsign)
        return R, Q

    def svd(self):
        u_blocks = [None] * self._nblocks
        s_blocks = [None] * self._nblocks
        v_blocks = [None] * self._nblocks
        for bi, b in enumerate(self._blocks):
            u_blocks[bi], s_blocks[bi], v_blocks[bi] = robust_svd(b)

        degen = np.array([s.size for s in s_blocks])
        mid_rep = self.init_representation(degen, self._block_irreps)
        usign = np.ones((self._nrr + 1,), dtype=bool)
        usign[: self._nrr] = self._signature[: self._nrr]
        U = type(self)(self._row_reps, (mid_rep,), u_blocks, self._block_irreps, usign)
        degens = [self.irrep_dimension(irr) for irr in self._block_irreps]
        s = DiagonalTensor(
            s_blocks, mid_rep, self._block_irreps, degens, self._symmetry
        )
        vsign = np.zeros((self._ndim - self._nrr + 1,), dtype=bool)
        vsign[1:] = self._signature[self._nrr :]
        V = type(self)((mid_rep,), self._col_reps, v_blocks, self._block_irreps, vsign)
        return U, s, V

    def eigh(self, *, compute_vectors=True):
        """
        Compute all eigenvalues and eigen of the SymmetricTensor viewed as a matrix.
        It has to be a square matrix, with same row_reps and col_reps and opposite
        signatures for rows and columns.
        Note that if self.blocks are not real symmetric or hermitian matrices, the
        results will be wrong.

        Parameters
        ----------
        compute_vectors : Bool
            Whether to compute and returns eigenvectors. Defaults to True.

        Returns
        -------
        s : DiagonalTensor
            eigenvalues as a DiagonalTensor. They define a new representation along the
            diagonal.
        u : SymmetricTensor
            Eigenvectors as a SymmetricTensor. Only returned if compute_vectors is
            True.

        Notes
        -----
        Exact zero eigenvalues may exist, especially when a block is missing. They will
        be truncated and no value or vector will be returned for the null eigenspace. In
        such as case, the dimensions of s and u will be smaller than self.
        """
        # just call eigsh with large nvals, it will call eigh on all blocks.
        # It would be possible to post-process eigenvalues and recover null eigenspace
        # Too complicate for no real use, just document behavior.
        return self._sparse_eig(
            self,
            2**31,
            None,
            None,
            None,
            2**31,
            compute_vectors,
            None,
            lambda b: robust_eigh(b, compute_vectors=compute_vectors),
            None,
        )

    def eig(self, *, compute_vectors=True):
        """
        Compute all eigenvalues and eigen of the SymmetricTensor viewed as a matrix.
        It has to be a square matrix, with same row_reps and col_reps and opposite
        signatures for rows and columns.

        Parameters
        ----------
        compute_vectors : Bool
            Whether to compute and returns eigenvectors. Defaults to True.

        Returns
        -------
        s : DiagonalTensor
            eigenvalues as a DiagonalTensor. They define a new representation along the
            diagonal.
        u : SymmetricTensor
            Eigenvectors as a SymmetricTensor. Only returned if compute_vectors is
            True.

        Notes
        -----
        Exact zero eigenvalues may exist, especially when a block is missing. They will
        be truncated and no value or vector will be returned for the null eigenspace. In
        such as case, the dimensions of s and u will be smaller than self.
        """
        # just call eigs with large nvals, it will call eig on all blocks.
        # It would be possible to post-process eigenvalues and recover null eigenspace
        # Too complicate for no real use, just document behavior.
        return self._sparse_eig(
            self,
            2**31,
            None,
            None,
            None,
            2**31,
            compute_vectors,
            None,
            lambda b: lg.eig(b, right=compute_vectors),
            None,
        )

    def expm(self):
        blocks = tuple(lg.expm(b) for b in self._blocks)
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, self._signature
        )

    ####################################################################################
    # Sparse linear algebra
    ####################################################################################
    def truncated_svd(
        self, cut, *, max_dense_dim=None, window=0, rcutoff=0.0, degen_ratio=1.0
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
                raw_u[bi], raw_s[bi], raw_v[bi] = robust_svd(b)
            else:
                raw_u[bi], raw_s[bi], raw_v[bi] = sparse_svd(b, cut + window)

        u_blocks = []
        s_blocks = []
        v_blocks = []
        dims = np.array([self.irrep_dimension(r) for r in self._block_irreps])
        block_cuts = find_chi_largest(
            raw_s, cut, dims=dims, rcutoff=rcutoff, degen_ratio=degen_ratio
        )
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
        usign = np.ones((self._nrr + 1,), dtype=bool)
        usign[: self._nrr] = self._signature[: self._nrr]
        U = type(self)(self._row_reps, (mid_rep,), u_blocks, block_irreps, usign)
        degens = [self.irrep_dimension(irr) for irr in block_irreps]
        s = DiagonalTensor(s_blocks, mid_rep, block_irreps, degens, self._symmetry)
        vsign = np.zeros((self._ndim - self._nrr + 1,), dtype=bool)
        vsign[1:] = self._signature[self._nrr :]
        V = type(self)((mid_rep,), self._col_reps, v_blocks, block_irreps, vsign)
        return U, s, V

    @classmethod
    def eigs(
        cls,
        matmat,
        nvals,
        *,
        reps=None,
        signature=None,
        dtype=None,
        dmax_full=100,
        compute_vectors=True,
        rng=None,
        maxiter=4000,
        tol=0,
    ):
        """
        Find nvals largest eigenvalues in magnitude for a square matrix M. M may be
        explicitly defined as a SymmetricTensor or only implicitly defined by its action
        on a vector with given representations, signature and dtype.
        Whether explicitly or implicitly defined, M has to define a square matrix
        that can act iteratively on a given initial vector. This means that its row
        representations must match its column representations and the signature for the
        columns must be the opposite of thw signature for the rows.

        Parameters
        ----------
        matmat : cls or callable
            If matmat is a SymmetricTensor with cls subclass then M=matmat and its
            spectrum will be direclty computed. Input arguments reps, signature and
            dtype are not read and are replaced by those inferred from matmat.
            If matmat is a callable, it represents the operation M @ x and implicitly
            defines a cls SymmetricTensor. Input arguments reps and signature are used
            to determine the vector it acts on.
        nvals : int
            Number of eigenvalues to compute. This number corresponds to dense
            eigenvalues, including multiplicites imposed by symmetry.
        reps : enumerable of representations
            Row representations for mat. Not read if mat is a SymmetricTensor. If mat
            is a callabel, reps also have to match mat column representations such that
            mat is a square  with same domain and codomain spaces.
        signature : bool 1D array
            Signature for mat rows. Not read if mat is a SymmetricTensor. Signature for
            mat column has to match ~signature.
        dtype : type
            Scalar data type for M. Not read if mat is a SymmetricTensor. Output dtype
            will always be np.complex128 regardless of this field.
        dmax_full : int
            Maximum block size to use dense eigvals.
        compute_vectors : Bool
            Whether to compute and returns eigenvectors. Default to True.
        rng : numpy random generator
            Random number generator. Used to initialize starting vectors for each block.
            If None, a new random generator is created with default_rng().
        maxiter : int
            Maximum number of Arnoldi iterations allowed in Arpack.
        tol : float
            Arpack tol.

        Returns
        -------
        s : DiagonalTensor
            eigenvalues as a DiagonalTensor. Final number of eigenvalues is the smallest
            number above nvals that fits multiplets.
        u : SymmetricTensor
            Eigenvectors as a SymmetricTensor. Only returned if compute_vectors is
            True.

        Notes
        -----
        Exact zero eigenvalues may exist, especially when a block is missing. They will
        be truncated and no value or vector will be returned for the null eigenspace.
        """

        def block_eigs(op, k, v0):
            return slg.eigs(
                op,
                k=k,
                v0=v0,
                maxiter=maxiter,
                tol=tol,
                return_eigenvectors=compute_vectors,
            )

        return cls._sparse_eig(
            matmat,
            nvals,
            reps,
            signature,
            dtype,
            dmax_full,
            compute_vectors,
            rng,
            lambda b: lg.eig(b, right=compute_vectors),
            block_eigs,
        )

    @classmethod
    def eigsh(
        cls,
        matmat,
        nvals,
        *,
        reps=None,
        signature=None,
        dtype=None,
        dmax_full=100,
        compute_vectors=True,
        rng=None,
        maxiter=4000,
        tol=0,
    ):
        """
        Find nvals largest eigenvalues in magnitude for a real symmetric or complex
        hermitian matrix M. M may be explicitly defined as a SymmetricTensor or only
        implicitly defined by its action on a vector with given representations,
        signature and dtype.
        Whether explicitly or implicitly defined, M has to define a square matrix
        that can act iteratively on a given initial vector. This means that its row
        representations must match its column representations and the signature for the
        columns must be the opposite of thw signature for the rows.

        If M is square but not real symmetric or hermitian, no error is returned but the
        results will be wrong.


        Parameters
        ----------
        matmat : cls or callable
            If matmat is a SymmetricTensor with cls subclass then M=matmat and its
            spectrum will be direclty computed. Input arguments reps, signature and
            dtype are not read and are replaced by those inferred from matmat.
            If matmat is a callable, it represents the operation M @ x and implicitly
            defines a cls SymmetricTensor. Input arguments reps and signature are used
            to determine the vector it acts on.
        nvals : int
            Number of eigenvalues to compute. This number corresponds to dense
            eigenvalues, including multiplicites imposed by symmetry.
        reps : enumerable of representations
            Row representations for mat. Not read if mat is a SymmetricTensor. If mat
            is a callable, reps also have to match mat column representations such that
            mat is a square  with same domain and codomain spaces.
        signature : bool 1D array
            Signature for mat rows. Not read if mat is a SymmetricTensor. Signature for
            mat column has to match ~signature.
        dtype : type
            Scalar data type for M. Not read if mat is a SymmetricTensor. Output dtype
            will always be np.float64 regardless of this field.
        dmax_full : int
            Maximum block size to use dense eigvals.
        compute_vectors : Bool
            Whether to compute and returns eigenvectors. Default to True.
        rng : numpy random generator
            Random number generator. Used to initialize starting vectors for each block.
            If None, a new random generator is created with default_rng().
        maxiter : int
            Maximum number of Lanczos update iterations allowed in Arpack.
        tol : float
            Arpack tol.

        Returns
        -------
        s : DiagonalTensor
            eigenvalues as a DiagonalTensor. Final number of eigenvalues is the smallest
            number above nvals that fits multiplets.
        u : SymmetricTensor
            Eigenvectors as a SymmetricTensor. Only returned if compute_vectors is
            True.

        Notes
        -----
        Exact zero eigenvalues may exist, especially when a block is missing. They will
        be truncated and no value or vector will be returned for the null eigenspace.
        """

        def block_eigs(op, k, v0):
            return slg.eigs(
                op,
                k=k,
                v0=v0,
                maxiter=maxiter,
                tol=tol,
                return_eigenvectors=compute_vectors,
            )

        return cls._sparse_eig(
            matmat,
            nvals,
            reps,
            signature,
            dtype,
            dmax_full,
            compute_vectors,
            rng,
            lambda b: robust_eigh(b, compute_vectors=compute_vectors),
            block_eigs,
        )

    ####################################################################################
    # I/O
    ####################################################################################
    def save_to_file(self, savefile):
        """
        Save SymmetricTensor into savefile with npz format.

        Save format may change to hdf5 in the future.
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
            prefix + "_symmetry": self._symmetry,
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
    def load_from_dic(cls, data, *, prefix=""):
        if cls._symmetry != data[prefix + "_symmetry"][()]:
            raise ValueError(f"Saved SymmetricTensor does not match type {cls}")
        nrr = int(data[prefix + "_n_row_reps"])
        row_reps = [data[f"{prefix}_row_rep_{ri}"] for ri in range(nrr)]
        nrc = int(data[prefix + "_n_col_reps"])
        col_reps = [data[f"{prefix}_col_rep_{ri}"] for ri in range(nrc)]
        block_irreps = data[prefix + "_block_irreps"]
        signature = data[prefix + "_signature"]
        blocks = [data[f"{prefix}_block_{bi}"] for bi in range(block_irreps.size)]
        return cls(row_reps, col_reps, blocks, block_irreps, signature)

    @classmethod
    def load_from_file(cls, savefile, *, prefix=""):
        with np.load(savefile) as fin:
            st = cls.load_from_dic(fin, prefix=prefix)
        return st

    ####################################################################################
    # Private methods
    ####################################################################################
    def _diagonal_mul(self, diag_blocks, diag_block_irreps, *, left=False):
        """
        Matrix product with a diagonal matrix with matching symmetry. If left is True,
        matrix multiplication is from the left.

        Convention: diag_blocks is understood as diagonal weights coming from a SVD in
        terms of representation and signature. Therefore it can only be added to the
        right if self has only one column leg and added to the left if self has only one
        Its signature is assumed to be trivial [False, True]. If it does not match self
        then a signature change is done by conjugating diag_block_irreps.

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

    @classmethod
    def _sparse_eig(
        cls,
        matmat,
        nvals,
        reps,
        signature,
        dtype,
        dmax_full,
        compute_vectors,
        rng,
        dense_eig,
        sparse_eig,
    ):
        """
        Private helper function. See eigsh documentation for parameters description.
        Other parameters:
        dense_eig: callable.
            Function for full diagonalization of a block, e.g. lambda b: lg.eigh(b)
        sparse_eig: callable.
            Function for sparse diagonalization of a block,
            e.g. lamba (op, k, v0): slg.eigsh(op, k=k, v0=v0)
        """
        # 0) input validation
        if type(matmat) is cls:
            nrr = matmat.n_row_reps
            if any(
                r1.shape != r2.shape or (r1 != r2).any()
                for (r1, r2) in zip(matmat.row_reps, matmat.col_reps, strict=True)
            ):
                raise ValueError(
                    "M representations are incompatible with a square matrix"
                )
            if (matmat.signature[:nrr] != ~matmat.signature[nrr:]).any():
                raise ValueError("M signature is incompatible with a square matrix")

            reps = matmat.row_reps
            signature = matmat.signature[:nrr]
            dtype = matmat.dtype

        elif callable(matmat):
            if reps is None or signature is None or dtype is None:
                raise ValueError(
                    "reps, signature and dtype must be specified for callable matmat"
                )
            nrr = len(reps)

        else:
            raise ValueError("Invalid input type for matmat")

        # 1) set parameters
        if rng is None:
            rng = np.random.default_rng()

        sigm = np.empty((2 * nrr,), dtype=bool)
        sigm[:nrr] = signature
        sigm[nrr:] = ~signature
        block_irreps, block_shapes = cls.get_block_sizes(reps, reps, sigm)
        nblocks = len(block_shapes)
        assert all(block_shapes[:, 0] == block_shapes[:, 1])
        sigv = np.ones((nrr + 1,), dtype=bool)
        sigv[:-1] = signature
        val_blocks = [None] * nblocks
        abs_val_blocks = [np.zeros((1,))] * nblocks  # avoid issue with missing block

        # 2) split matrix blocks between full and sparse
        sparse = []
        full = []
        if type(matmat) is not cls:
            dense_blocks = []
        dims = np.empty((nblocks,), dtype=int)
        for bi in range(nblocks):
            irr = block_irreps[bi]
            dims[bi] = cls.irrep_dimension(irr)
            d = block_shapes[bi, 0]  # size of the block in symmetric format
            k = nvals // dims[bi] + 1  # number of eigenvalues to compute in this block
            if d < max(dmax_full, 3 * k):  # small blocks: dense
                full.append(bi)
                if type(matmat) is not cls:
                    dense_blocks.append(np.eye(d, dtype=dtype))
            else:
                sparse.append(bi)

        # 3) define functions do deal with dense and sparse blocks
        def matvec(x, st0, bj):
            st0.blocks[0][:, 0] = x
            st1 = matmat(st0)
            y = st1.blocks[bj].ravel()
            return y

        if compute_vectors:
            vector_blocks = [None] * nblocks

            def eig_full_block(bi, b, k):
                vals, vec = dense_eig(b)
                abs_val = np.abs(vals)
                so = abs_val.argsort()[: -k - 1 : -1]
                val_blocks[bi] = vals[so]
                abs_val_blocks[bi] = abs_val[so]
                vector_blocks[bi] = vec[:, so]

            def eig_sparse_block(bi, op, k, v0):
                try:
                    vals, vec = sparse_eig(op, k, v0)
                except slg.ArpackNoConvergence as err:
                    print("Warning: ARPACK did not converge", err)
                    vals = err.eigenvalues
                    vec = err.eigenvectors
                    print(f"Keep {vals.size}/{k} converged values and vectors")

                abs_val = np.abs(vals)
                so = abs_val.argsort()[: -k - 1 : -1]
                val_blocks[bi] = vals[so]
                abs_val_blocks[bi] = abs_val[so]
                vector_blocks[bi] = vec[:, so]

        else:

            def eig_full_block(bi, b, k):
                vals = dense_eig(b)
                abs_val = np.abs(vals)
                so = abs_val.argsort()[: -k - 1 : -1]
                val_blocks[bi] = vals[so]
                abs_val_blocks[bi] = abs_val[so]

            def eig_sparse_block(bi, op, k, v0):
                try:
                    vals = sparse_eig(op, k, v0)
                except slg.ArpackNoConvergence as err:
                    print("Warning: ARPACK did not converge", err)
                    vals = err.eigenvalues
                    print(f"Keep {vals.size}/{k} converged eigvalues")

                abs_val = np.abs(vals)
                so = abs_val.argsort()[: -k - 1 : -1]
                val_blocks[bi] = vals[so]
                abs_val_blocks[bi] = abs_val[so]

        # 4) construct full matrix blocks and call dense eigh on them
        if type(matmat) is cls:
            st = matmat  # use already constructed blocks
        elif full:  # avoid issues with empty full
            # use just one call of matmat on identity blocks to produce all blocks
            irr_full = np.ascontiguousarray(block_irreps[full])
            rfull = cls.init_representation(block_shapes[full, 0], irr_full)
            st = cls(reps, (rfull,), dense_blocks, irr_full, sigv)
            st = matmat(st)
        for bi in full:
            k = nvals // dims[bi] + 1
            irr = block_irreps[bi]
            bj = st.block_irreps.searchsorted(irr)
            if bj < st.nblocks and st.block_irreps[bj] == irr:
                eig_full_block(bi, st.blocks[bj], k)
            # else the block is missing, truncate zero eigenvalue

        # 5) for each sparse block, apply matmat to a SymmetricTensor with 1 block
        for bi in sparse:
            irr = block_irreps[bi]
            sh = block_shapes[bi]
            k = nvals // dims[bi] + 1
            v0 = rng.normal(size=(sh[0],)).astype(dtype, copy=False)
            if type(matmat) is cls:  # use constructed blocks
                bj = matmat.block_irreps.searchsorted(irr)
                if bj < matmat.nblocks and matmat.block_irreps[bj] == irr:
                    op = matmat.blocks[bj]
                    eig_sparse_block(bi, op, k, v0)
                # else the block is missing

            else:
                block_irreps_bi = block_irreps[bi : bi + 1]
                brep = cls.init_representation(
                    np.ones((1,), dtype=int), block_irreps_bi
                )

                st0 = cls(reps, (brep,), (v0[:, None],), block_irreps_bi, sigv)
                st1 = matmat(st0)
                # assume bj does not depend on x values and stays fixed over iterations
                bj = st1.block_irreps.searchsorted(irr)
                op = slg.LinearOperator(
                    sh, matvec=lambda x: matvec(x, st0, bj), dtype=dtype  # noqa: B023
                )

                # check that irr block actually appears in output
                if bj < st1.nblocks and st1.block_irreps[bj] == irr:
                    eig_sparse_block(bi, op, k, v0)

        # 6) keep only nvals largest magnitude eigenvalues
        # find_chi_largest will remove any rigorously zero eigenvalue
        block_cuts = find_chi_largest(abs_val_blocks, nvals, dims=dims)
        non_empty = block_cuts.nonzero()[0]
        s_blocks = []
        for bi in non_empty:
            bcut = block_cuts[bi]
            s_blocks.append(val_blocks[bi][:bcut])

        # 7) construct DiagonalTensor for the eigenvalues
        block_irreps = block_irreps[non_empty]
        s_rep = cls.init_representation(block_cuts[non_empty], block_irreps)
        degens = [cls.irrep_dimension(irr) for irr in block_irreps]
        s = DiagonalTensor(s_blocks, s_rep, block_irreps, degens, cls._symmetry)

        if not compute_vectors:
            return s

        # 8) construct SymmetricTensor for the eigenvectors
        u_blocks = []
        for bi in non_empty:
            bcut = block_cuts[bi]
            u_blocks.append(vector_blocks[bi][:, :bcut])

        mid_rep = cls.init_representation(block_cuts[non_empty], block_irreps)
        usign = np.ones((nrr + 1,), dtype=bool)
        usign[:nrr] = signature
        u = cls(reps, (mid_rep,), u_blocks, block_irreps, usign)
        return s, u
