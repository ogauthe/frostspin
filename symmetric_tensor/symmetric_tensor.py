import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg

from misc_tools.svd_tools import sparse_svd, find_chi_largest
from symmetric_tensor.diagonal_tensor import DiagonalTensor

if __debug__:
    print("\nWarning: assert statement are activated")
    print("They may significantly impact performances")
    print("Consider running the code in optimized mode with python -O")
    print()

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

    def toarray(self, as_matrix=False):
        raise NotImplementedError("Must be defined in derived class")

    def permute(self, row_axes, col_axes):  # signature != ndarray.transpose
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
        if isinstance(other, DiagonalTensor):  # add diagonal weights on 1st row leg
            assert self._symmetry == other.symmetry()
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
        assert type(self) is type(other)
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
        if type(self) is not type(other):
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
        Matrix transpose operation, swapping rows and columns. Internal structure and
        and leg merging are not affected: each block is just transposed. Block irreps
        are group conjugated, which may change block order.
        """
        raise NotImplementedError("Must be defined in derived class")

    def dual(self):
        """
        Return a new tensor with all representations (row, columns and blocks irreps)
        conjugated according to group rules. Since the dense tensor is a group singlet,
        it is unaffected in its dense form, however symmetric blocks may change,
        especially in the non-abelian case.
        """
        raise NotImplementedError("Must be defined in derived class")

    def conjugate(self):
        """
        Complex conjugate operation. Block values are conjugate and all representations
        are mapped to their dual.
        """
        conj = self.dual()
        conj._blocks = tuple(b.conj() for b in conj._blocks)
        return conj

    def dagger(self):
        """
        Hermitian conjugate operation, swapping rows and columns and conjugating blocks.
        block_irreps and block order are not affected.
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
        for irr, b in zip(self._block_irreps, self._blocks):
            n2 += self.irrep_dimension(irr) * lg.norm(b) ** 2
        return np.sqrt(n2)

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

    def expm(self):
        blocks = tuple(lg.expm(b) for b in self._blocks)
        return type(self)(
            self._row_reps, self._col_reps, blocks, self._block_irreps, self._signature
        )

    ####################################################################################
    # Sparse linear algebra
    ####################################################################################
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
        mat,
        nvals,
        *,
        reps=None,
        signature=None,
        dtype=None,
        dmax_full=100,
        rng=None,
        maxiter=4000,
        tol=0,
    ):
        """
        Find nvals eigenvalues for a square matrix M. M is only implicitly defined by
        its action on a vector with fixed signature and representations.

        Parameters
        ----------
        mat : cls or callable
            If mat is a SymmetricTensor with cls subclass, it is the matrix whose
            spectrum will be computed. Input arguments reps, signature and dtype are
            not read and are replaced by those infered from mat.
            If mat is a callable, it represents the operation mat @ x and implicitly
            defines a cls SymmetricTensor. Input arguments reps and signature are used
            to determine the vector it acts on.
            Whether explicitly or implicitly defined, mat has to be a square matrix
            that can act iteratively on a given initial vector.
            SymmetricTensor st, then lambda x: st @ x is used. Representation and
            signature have to match those in reps and signature.
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
            Scalar data type. Not read if mat is a SymmetricTensor. Else default to
            np.complex128.
        dmax_full : int
            Maximum block size to use dense eigvals.
        rng : numpy random generator
            Random number generator. Used to initialize starting vectors for each block.
            If None, a new random generator is created with default_rng().
        maxiter : int
            Maximum number of Arnoldi update iterations allowed in Arpack.
        tol : float
            Arpack tol.

        Returns
        -------
        s : DiagonalTensor
            eigenvalues as a DiagonalTensor. Final number of eigenvalues is the smallest
            number above nvals that fits multiplets.
        """

        # 0) input validation
        if type(mat) is cls:
            n = mat.n_row_reps
            if mat.shape[:n] != mat.shape[n:]:
                raise ValueError("mat shape incompatible with a square matrix")
            if any(
                r1.shape != r2.shape or (r1 != r2).any()
                for (r1, r2) in zip(mat.row_reps, mat.col_reps)
            ):
                raise ValueError(
                    "mat representations incompatible with a square matrix"
                )
            if (mat.signature[:n] != ~mat.signature[n:]).any():
                raise ValueError("mat signature incompatible with square matrix")

            reps = mat.row_reps
            signature = mat.signature[:n]
            dtype = mat.dtype
            # could be slightly more efficient by using already constructed matrix
            # blocks. Only small gain expected, not worth code dupplication.

            def matmat(x):
                return mat @ x

        elif callable(mat):
            matmat = mat
            if reps is None or signature is None:
                raise ValueError(
                    "reps and signature must be specified for callable mat"
                )
            n = len(reps)

        else:
            raise ValueError("Invalid input mat")

        # 1) set parameters
        # if dtype is real, most of the computation can be done with reals
        # however eigvals will always produce complex
        # so at some point need to forget dtype and use complex128 for return type
        if dtype is None:
            dtype = np.complex128
        if rng is None:
            rng = np.random.default_rng()

        sigm = np.empty((2 * n,), dtype=bool)
        sigm[:n] = signature
        sigm[n:] = ~signature
        block_irreps, block_shapes = cls.get_block_sizes(reps, reps, sigm)
        nblocks = len(block_shapes)
        assert all(block_shapes[:, 0] == block_shapes[:, 1])
        sigv = np.ones((n + 1,), dtype=bool)
        sigv[:-1] = signature
        ev_blocks = [None] * nblocks
        abs_ev_blocks = [None] * nblocks

        # 2) split matrix blocks between full and sparse
        sparse = []
        full = []
        blocks = []
        dims = np.empty((nblocks,), dtype=int)
        for bi in range(nblocks):
            irr = block_irreps[bi]
            dims[bi] = cls.irrep_dimension(irr)
            if block_shapes[bi, 0] > max(dmax_full, 2 * nvals / dims[bi]):
                sparse.append(bi)
            else:
                full.append(bi)
                blocks.append(np.eye(block_shapes[bi, 0], dtype=dtype))

        # 3) construct full matrix blocks and call dense eig on them
        # use just one call of matmat on identity blocks to produce all blocks
        if full:
            irr_full = np.ascontiguousarray(block_irreps[full])
            rfull = cls.init_representation(block_shapes[full, 0], irr_full)
            st0 = cls(reps, (rfull,), blocks, irr_full, sigv)
            st1 = matmat(st0)
            for bi in full:
                irr = block_irreps[bi]
                bj = st1.block_irreps.searchsorted(irr)
                if bj < st1.nblocks and st1.block_irreps[bj] == irr:
                    ev = lg.eigvals(st1.blocks[bj])
                    abs_ev = np.abs(ev)
                    so = abs_ev.argsort()[::-1]
                    ev_blocks[bi] = ev[so]
                    abs_ev_blocks[bi] = abs_ev[so]
                else:  # missing block means eigval = 0
                    ev_blocks[bi] = np.zeros(
                        (block_shapes[bi, 0],), dtype=np.complex128
                    )
                    abs_ev_blocks[bi] = np.zeros((block_shapes[bi, 0],))

        # 4) for each sparse block, apply matmat to a SymmetricTensor with 1 block
        for bi in sparse:
            irr = block_irreps[bi]
            block_irreps_bi = block_irreps[bi : bi + 1]
            brep = cls.init_representation(np.ones((1,), dtype=int), block_irreps_bi)
            sh = block_shapes[bi]

            v0 = rng.normal(size=(sh[0],)).astype(dtype, copy=False)
            st0 = cls(reps, (brep,), (v0[:, None],), block_irreps_bi, sigv)
            st1 = matmat(st0)
            bj = st1.block_irreps.searchsorted(irr)

            # check that irr block actually appears in output
            if bj < st1.nblocks and st1.block_irreps[bj] == irr:

                def matvec(x):
                    st0.blocks[0][:, 0] = x
                    st1 = matmat(st0)
                    # here we assume bj does not depend on x values
                    y = st1.blocks[bj].ravel()
                    return y

                op = slg.LinearOperator(sh, matvec=matvec, dtype=dtype)
                k = nvals // dims[bi] + 1
                try:
                    ev = slg.eigs(
                        op,
                        k=k,
                        v0=v0,
                        maxiter=maxiter,
                        tol=tol,
                        return_eigenvectors=False,
                    )
                except slg.ArpackNoConvergence as err:
                    print("Warning: ARPACK did not converge", err)
                    ev = err.eigenvalues
                    print(f"Keep {ev.size} converged eigenvalues")

                abs_ev = np.abs(ev)
                so = abs_ev.argsort()[::-1]
                ev_blocks[bi] = ev[so]
                abs_ev_blocks[bi] = abs_ev[so]

            else:  # missing block
                ev_blocks[bi] = np.zeros((nvals,), dtype=np.complex128)
                abs_ev_blocks[bi] = np.zeros((nvals,))

        block_cuts = find_chi_largest(abs_ev_blocks, nvals, dims=dims)
        non_empty = block_cuts.nonzero()[0]
        s_blocks = []
        for bi in non_empty:
            bcut = block_cuts[bi]
            s_blocks.append(ev_blocks[bi][:bcut])

        block_irreps = block_irreps[non_empty]
        s_rep = cls.init_representation(block_cuts[non_empty], block_irreps)
        degens = [cls.irrep_dimension(irr) for irr in block_irreps]
        s = DiagonalTensor(s_blocks, s_rep, block_irreps, degens, cls._symmetry)
        return s

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
    def load_from_dic(cls, data, prefix=""):
        if cls._symmetry != data[prefix + "_symmetry"][()]:
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
