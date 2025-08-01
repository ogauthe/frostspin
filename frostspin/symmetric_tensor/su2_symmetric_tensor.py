import numba
import numpy as np

from frostspin.groups.su2_clebsch_gordan import load_su2_cg

from .lie_group_symmetric_tensor import LieGroupSymmetricTensor
from .o2_symmetric_tensor import O2SymmetricTensor
from .tools import check_norm, symmetric_tensor_types
from .u1_symmetric_tensor import U1SymmetricTensor


def compute_fusion_tensor(rep1, rep2, s1, s2, *, max_irrep=2**30):
    """
    Construct Clebsch-Gordan fusion tensor for representations rep1 and rep2 with
    signatures s1 and s2.

    Parameters
    ----------
    rep1 : 2D int array
        Left incoming SU(2) representation to fuse.
    rep2 : 2D int array
        Right incoming SU(2) representation to fuse.
    s1 : bool
        Signature for rep1.
    s2 : bool
        Signature for rep2.
    max_irrep : int
        Dimension of maximal irrep to consider in the product. Irrep larger than
        max_irrep will be truncated. Default is 2**30, i.e. no truncation.

    Returns
    -------
    ret : 3D float array
        CG projector fusing rep1 and rep2 on sum of irreps, truncated up to max_irrep.

    Notes
    -----
        The output matrix hides a structure
        (degen1, irrep1, degen2, irrep2, degen1, degen2, irrep3).
        This is not exactly a tensor but corresponds to how row and columns relates to
        degeneracies and irreducible representations.
    """
    # this is function is specific to SU(2), we can safely assume an irrep is labeled by
    # its dimension and use label or dimension indistinctly.
    cg_dic = SU2SymmetricTensor.clebsch_gordan_dic()

    # precompute projector size
    degen, irreps = _numba_elementary_combine_SU2(rep1[0], rep1[1], rep2[0], rep2[1])
    trunc = irreps.searchsorted(max_irrep + 1)
    degen = degen[:trunc]
    irreps = irreps[:trunc]
    projector = np.zeros(
        (rep1[0] @ rep1[1], rep2[0] @ rep2[1], degen @ irreps),
    )
    if trunc == 0:  # no irrep allowed
        return np.array([[], []], dtype=int), projector

    # initialize shifts in output, with different irrep sectors
    shifts3 = np.empty((irreps[-1] + 1,), dtype=int)
    n = 0
    for d3, irr3 in zip(degen, irreps, strict=True):
        shifts3[irr3] = n  # indexed with IRREP, not index (shortcut searchsorted)
        n += d3 * irr3

    shift1 = 0
    for d1, irr1 in rep1.T:
        # Sz-reversal signs for irrep1
        diag1 = (np.arange(irr1 % 2, irr1 + irr1 % 2) % 2 * 2 - 1)[:, None, None]
        shift2 = 0
        for d2, irr2 in rep2.T:
            # Sz-reversal signs for irrep2
            diag2 = (np.arange(irr2 % 2, irr2 + irr2 % 2) % 2 * 2 - 1)[None, :, None]
            for irr3 in range(abs(irr1 - irr2) + 1, min(irr1 + irr2, max_irrep + 1), 2):
                # here we implicitly make use of the fact SU(2) has no outer degeneracy
                try:
                    p123 = cg_dic[irr1, irr2, irr3]
                except KeyError:
                    msg = (
                        "\n*** ERROR *** Clebsch-Gordan tensor for spins with "
                        f"dimensions {(irr1, irr2, irr3)} not found.\nOn the fly"
                        "computation of CG tensors is not currently implemented.\nTo "
                        "compute a larger set of CG tensors, use the script "
                        "frostspin/groups/compute_su2_clebsch_gordan.py.\n"
                    )
                    raise RuntimeError(msg) from None

                # apply SU(2) spin-reversal operator according to signatures
                if s1:
                    p123 = p123[::-1] * diag1
                if s2:
                    p123 = p123[:, ::-1] * diag2

                for i3 in range(d1 * d2):
                    i1, i2 = divmod(i3, d2)
                    sl1 = slice(shift1 + i1 * irr1, shift1 + (i1 + 1) * irr1)
                    sl2 = slice(shift2 + i2 * irr2, shift2 + (i2 + 1) * irr2)
                    sl3 = slice(
                        shifts3[irr3] + i3 * irr3, shifts3[irr3] + (i3 + 1) * irr3
                    )
                    projector[sl1, sl2, sl3] = p123
                shifts3[irr3] += d1 * d2 * irr3
            shift2 += d2 * irr2
        shift1 += d1 * irr1

    new_rep = np.array([degen, irreps])  # due to truncation, may be != rep1 * rep2
    return new_rep, projector


@numba.njit
def _numba_elementary_combine_SU2(degen1, irreps1, degen2, irreps2):
    degen = np.zeros(irreps1[-1] + irreps2[-1] - 1, dtype=np.int64)
    for i1, irr1 in enumerate(irreps1):
        d1 = degen1[i1]
        for i2, irr2 in enumerate(irreps2):
            d2 = degen2[i2]
            for irr in range(abs(irr1 - irr2), irr1 + irr2 - 1, 2):
                degen[irr] += d1 * d2  # shit irr-1 <-- irr to start at 0
    nnz = degen.nonzero()[0]
    return degen[nnz], nnz + 1


@numba.njit
def _numba_combine_SU2(*reps):
    degen, irreps = reps[0]
    for r in reps[1:]:
        degen, irreps = _numba_elementary_combine_SU2(degen, irreps, r[0], r[1])
    return np.concatenate((degen, irreps)).reshape(2, -1)


class SU2SymmetricTensor(LieGroupSymmetricTensor):
    """
    Irreps are 2D arrays with int dtype. First row is degen, second row is irrep
    dimension = 2 * s + 1
    """

    ####################################################################################
    # Symmetry implementation
    ####################################################################################
    _symmetry = "SU2"
    _clebsch_gordan_dic = load_su2_cg()

    @classmethod
    def clebsch_gordan_dic(cls):
        return cls._clebsch_gordan_dic

    @staticmethod
    def singlet():
        return np.ones((2, 1), dtype=int)

    @staticmethod
    def combine_representations(reps, _signature):
        if len(reps) > 1:  # numba issue 7245
            return _numba_combine_SU2(*reps)
        return reps[0]

    @staticmethod
    def conjugate_irrep(irr):
        return irr

    @staticmethod
    def conjugate_representation(rep):
        return rep

    @staticmethod
    def representation_dimension(rep):
        return rep[0] @ rep[1]

    @staticmethod
    def irrep_dimension(irr):
        return int(irr)

    ####################################################################################
    # Non-abelian specific symmetry implementation
    ####################################################################################
    _structural_data_dic = {}
    _unitary_dic = {}

    @classmethod
    def compute_clebsch_gordan_tree(cls, rep_in, signature, *, target_irreps=None):
        assert len(signature) == len(rep_in)

        n = len(rep_in)
        if n == 1:
            # add a dummy singlet and let get_projector deal with signature
            rep_in = [rep_in[0], np.ones((2, 1), dtype=int)]
            signature = [signature[0], False]
            n = 2

        # remove irreps that wont fuse to max_irrep
        # too complicate to prune all non-contributing irreps at every stage
        # just prune spins larger than max_spin(product spins left)
        # should give the same pruning in most cases
        if target_irreps is None:
            max_irrep = 2**30
        else:
            max_irrep = target_irreps[-1] + sum(r[1, -1] for r in rep_in[2:]) - n + 2

        nr, p = compute_fusion_tensor(
            rep_in[0], rep_in[1], signature[0], signature[1], max_irrep=max_irrep
        )
        if nr.size == 0:  # pathological case where no irrep is kept
            return nr, np.zeros((p.shape[0] * p.shape[1], 0))

        proj = p
        for i in range(2, n):
            max_irrep -= rep_in[i][1, -1] - 1
            nr, p = compute_fusion_tensor(
                nr, rep_in[i], False, signature[i], max_irrep=max_irrep  # noqa: FBT003
            )
            if nr.size == 0:  # pathological case where no irrep is kept
                return nr, np.zeros((p.shape[0] * p.shape[1], 0))
            proj = proj.reshape(-1, p.shape[0]) @ p.reshape(p.shape[0], -1)
        proj = proj.reshape(-1, p.shape[2])
        return nr, proj

    ####################################################################################
    # Symmetry specific methods with fixed signature
    ####################################################################################
    def toSU2(self):
        return self

    def toabelian(self):
        return self.toU1()

    def toU1(self):
        # this is not efficient: cast to dense, then back to U(1)
        # not worth it to optimize.
        reps = []
        for r in self._row_reps + self._col_reps:
            sz = np.empty((r[0] @ r[1],), dtype=np.int8)
            k = 0
            for d, irr in np.nditer(r, flags=["external_loop"], order="F"):
                sz_irr = np.arange(irr - 1, -irr - 1, -2, dtype=np.int8)
                sz[k : k + d * irr].reshape(d, irr)[:] = sz_irr
                k += d * irr
            reps.append(sz)

        arr = self.toarray()
        out = U1SymmetricTensor.from_array(
            arr, reps[: self._nrr], reps[self._nrr :], self._signature
        )
        assert check_norm(self, out)
        return out

    def toO2(self):
        """
        Warning: this method alters the dense tensors by swapping indices inside legs
        and adding some diagonal -1 signs on every legs. This does not matter once legs
        are contracted.
        """
        # this is not efficient: cast to dense, then back to U(1)
        # not worth it to optimize.

        # When casting to U(1), O(2) has different irrep ordering conventions:
        # here for SU(2), each spin appears contiguously with all its Sz value
        # For O(2), dim 2 irreps are mixed to get a contiguous +n sector and a
        # contiguous -n sector
        # e.g. for 2 spins 1, SU(2).toU1() gives Sz = [2,0,-2,2,0,-2]
        # while for O(2), this gives 2 0odd and 2 irreps 2 with Sz = [0,0,2,2,-2,-2]
        # hence some permutation is needed for every leg.

        # Furthermore, SU(2) follows standard spin-reversal convention, with vector |Sz>
        # being map to +/-|-Sz> depending on irrep.
        # For O(2), we impose even Z|Sz> -> |-Sz> for integer Sz (even with factor 2)
        # Z|Sz> -> +|-Sz> for Sz > 0 and Z|Sz> -> -|-Sz> for Sz < 0 for half interger Sz
        # This means starting from spin 3/2, some basis vector must be redefined with a
        # -1 sign, which introduces signs in the tensor coefficients.

        # reuse U(1) code in spirit
        # 1) construct U(1) and O(2) representations
        swaps = []  # SU(2) and O(2) have different ordering convention: swap needed
        o2_reps = []
        signs = []  # change vector signs to fit O(2) signs conventions
        for r in self._row_reps + self._col_reps:
            sz_rep = np.empty((r[0] @ r[1],), dtype=np.int8)
            sz_rep_o2 = np.empty(sz_rep.shape, dtype=np.int8)
            k = 0
            signs_rep = np.empty(sz_rep.shape, dtype=bool)
            for d, irr in np.nditer(r, flags=["external_loop"], order="F"):
                sz_irr = np.arange(irr - 1, -irr - 1, -2, dtype=np.int8)
                sz_rep[k : k + d * irr].reshape(d, irr)[:] = sz_irr
                sz_irr_o2 = np.abs(sz_irr)
                signs_irr = np.zeros(sz_irr.shape, dtype=bool)
                if irr % 2:
                    signs_irr[(sz_irr % 4 == (irr + 1) % 4) & (sz_irr < 0)] = True
                    if irr % 4 == 3:  # odd integer spin -> Sz=0 is odd
                        sz_irr_o2[irr // 2] = -1
                else:
                    signs_irr[(sz_irr % 4 == (irr - 1) % 4) & (sz_irr < 0)] = True
                sz_rep_o2[k : k + d * irr].reshape(d, irr)[:] = sz_irr_o2
                signs_rep[k : k + d * irr].reshape(d, irr)[:] = signs_irr
                k += d * irr
            swap_rep = (-sz_rep).argsort(kind="stable")
            swap_rep = swap_rep[sz_rep_o2[swap_rep].argsort(kind="stable")]
            signs.append(1 - 2 * signs_rep)
            swaps.append(swap_rep)  # store swap
            irreps, degen = np.unique(sz_rep_o2, return_counts=True)
            degen[irreps > 0] //= 2
            o2_reps.append(np.array([degen, irreps]))

        # 2) combine into row and column
        arr = self.toarray()
        perm = (*range(1, self._ndim), 0)
        for ax in range(self._ndim):
            mat = np.eye(self._shape[ax])
            mat = (mat * signs[ax])[swaps[ax]]
            arr = np.tensordot(mat, arr, ((1,), (0,)))
            arr = arr.transpose(perm)

        out = O2SymmetricTensor.from_array(
            arr, o2_reps[: self._nrr], o2_reps[self._nrr :], self._signature
        )
        assert check_norm(self, out)
        return out

    ####################################################################################
    # SU(2) specific
    ####################################################################################
    @classmethod
    def reload_clebsch_gordan(cls, savefile, *, verbosity=0):
        """
        Reload SU(2) Clebsch-Gordan cofficient dictionary from savefile. Coefficients
        are loaded at import, this function allows to reload them from another file.

        Arguments
        ---------
        savefile : str
            Savefile for SU(2) Clebsch-Gordan coefficients. Must be a .json or .npz file
            and abide by the format defined in frostspin/groups/su2_clebsch_gordan.py.
        """
        cls._clebsch_gordan_dic = load_su2_cg(savefile, verbosity=verbosity)


symmetric_tensor_types["SU2"] = SU2SymmetricTensor
