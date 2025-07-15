import numpy as np

from frostspin import DiagonalTensor
from frostspin.symmetric_tensor.tools import get_symmetric_tensor_type

from .abstract_simple_update import AbstractSimpleUpdate
from .simple_update_tools import check_hamiltonians


class SymmetrizedSimpleUpdate(AbstractSimpleUpdate):
    _initial_swap = [((1, 2, 4, 5), (3, 0))]
    _final_swap = [((5, 0), (4, 1, 2, 3))]
    _lperm = [
        ((0, 4, 2, 3), (1, 5)),
        ((0, 4, 2, 3), (1, 5)),
        ((0, 1, 4, 3), (2, 5)),
        ((0, 1, 2, 4), (3, 5)),
        ((0, 1, 2, 4), (3, 5)),
        ((0, 1, 4, 3), (2, 5)),
        ((0, 4, 2, 3), (1, 5)),
    ]
    _tensor_bond_indices = np.array([[0, 1, 2, 3]])
    _1st_updated_bond = [0, 1, 2, 3, 2, 1, 0]
    _gate_indices = [1, 0, 0, 2, 0, 0, 0]
    _raw_update_data = np.zeros((0, 0), dtype=int)  # only used in abstract save_to_file
    _is_second_order = True
    _n_bonds = 4
    _n_updates = 6
    _n_tensors = 1
    _classname = "SymmetrizedSimpleUpdate"  # used in save/load to check consistency

    def __repr__(self):
        s = f"SymmetrizedSimpleUpdate with {self._symmetry} symmetry and D = {self.D}"
        return s + f" at beta = {self._beta:.10g}"

    def __init__(
        self,
        D,
        beta,
        tau,
        rcutoff,
        degen_ratio,
        A,
        raw_hamilts,
        weights,
        logZ,
        *,
        verbosity=0,
    ):
        self.verbosity = int(verbosity)

        # quick crash for very simple errors
        if len(weights) != 4:
            raise ValueError("Invalid weight length")
        check_hamiltonians(raw_hamilts)

        self._beta = beta
        self._symmetry = raw_hamilts[0].symmetry()
        self._tensors = [A]
        self._weights = list(weights)
        self.D = D
        self._logZ = logZ
        self.rcutoff = rcutoff
        self.degen_ratio = degen_ratio
        self._raw_hamilts = raw_hamilts

        if len(raw_hamilts) == 1:
            self._hamiltonians = [
                raw_hamilts[0],
                2 * raw_hamilts[0],
                2 * raw_hamilts[0],
            ]
        elif len(raw_hamilts) == 4:
            self._hamiltonians = [
                raw_hamilts[0],
                raw_hamilts[1],
                raw_hamilts[2],
                2 * raw_hamilts[3],
                2 * raw_hamilts[0],
            ]
        else:
            raise ValueError("Invalid raw_hamiltonians")
        self.tau = tau  # also set gates

    @property
    def logZ(self):
        return self._logZ / 2  # here need n_tensors = 2

    @classmethod
    def from_infinite_temperature(
        cls,
        D,
        tau,
        raw_hamilts,
        *,
        rcutoff=1e-14,
        degen_ratio=0.99999,
        verbosity=0,
    ):
        """
        Initialize finite temperature SimpleUpdate at beta = 0 eact product state.

        Parameters
        ----------
        D : int
            Bond dimension to keep when renormalizing bonds. This is a target, the
        actual largest value Dmax may differ due to cutoff or degeneracies.
        tau : float
            Imaginary time step.
        raw_hamilts : enumerable of SymmetricTensor
            List of elementary bond Hamiltonians acting on the tensors.
        rcutoff : float
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        degen_ratio : float
            Consider singular values degenerate if their quotient is above degen_ratio.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        n_bonds = 4
        ST = type(raw_hamilts[0])

        # initalize tensors
        phys_rep = raw_hamilts[0].row_reps[0]
        d = ST.representation_dimension(phys_rep)
        mat = np.eye(d).reshape((d, d, 1, 1, 1, 1))
        row_reps = (phys_rep, phys_rep)
        col_reps = (ST.singlet(),) * n_bonds
        assert raw_hamilts[0].signature[0] ^ raw_hamilts[0].signature[1]
        s0 = raw_hamilts[0].signature[0]
        a0 = ST.from_array(mat, row_reps, col_reps, signature=[~s0, s0, 1, 1, 1, 1])

        beta = 0.0
        irr = a0.block_irreps
        w = [np.ones((1,))]
        weights = [
            DiagonalTensor(w, ST.singlet(), irr, [1], ST.symmetry())
            for _ in range(n_bonds)
        ]
        logZ = 0.0

        return cls(
            D,
            beta,
            tau,
            rcutoff,
            degen_ratio,
            a0,
            raw_hamilts,
            weights,
            logZ,
            verbosity=verbosity,
        )

    def get_tensors(self):
        """
        Returns
        -------
        tensors : tuple of _n_tensors SymmetricTensor
            Optimized tensors, with sqrt(weights) on all virtual legs.
        """
        # we already imposed the two first legs to be physical and ancilla in the
        # default configuration. Add weights on the virtual legs.
        t = self._tensors[0]
        for i in range(4):
            t = t.permute((0, 1, 3, 4, 5), (2,))
            t = t * self._weights[i] ** -0.5
        t = t.permute((0, 1), (2, 3, 4, 5))
        return [t, t.dagger().permute((4, 5), (2, 3, 0, 1))]

    def update_first_neighbor(self, left, weights, gate):
        r""" """
        # cut left and right between const and effective parts
        cstL, effL = left.qr()  # auxL-effL=m,p

        # change tensor structure to contract mid
        effL = effL.permute((0, 2), (1,)) * weights ** (-1 / 2)  # auxL,p=effL-m

        # construct matrix theta and apply gate
        theta = effL @ effL.dagger()  # auxL,pL=theta=auxR,pR
        theta = theta.permute((0, 2), (1, 3))  # auxL, auxR = theta = pL, pR
        theta = theta @ gate

        # transpose back LxR, compute eigh and truncate
        theta = theta.permute((0, 2), (1, 3))  # auxL, pL = theta = auxR, pR
        # define new_weights *on effL right*
        new_weights, new_effL = theta.eigsh(
            theta, self.D, rtol=self.rcutoff, degen_ratio=self.degen_ratio
        )

        # normalize weights and apply them to new left and new right
        # save log of normalization factor to update logZ
        new_weights /= new_weights.sum()
        lognf = np.log(theta.norm() / new_weights.norm())
        new_effL = new_effL * new_weights

        # reshape to initial tree structure
        new_effL = new_effL.permute((0,), (2, 1))  # auxL - effL = m,pL

        # reconnect with const parts
        newL = cstL @ new_effL

        return newL, new_weights, lognf

    def _elementary_update(self, i):
        """
        Elementary update bond between tensors iL and iR.
        """
        # This function is made to be fast: assume current state is fine and use
        # precomputed indices to select bonds, kind of update, tensors as well as
        # precomputed leg permutation.
        left = self._tensors[0].permute(*self._lperm[i])
        b1 = self._1st_updated_bond[i]
        new_left, nw1, lognf = self.update_first_neighbor(
            left, self._weights[b1], self._gates[self._gate_indices[i]]
        )
        self._logZ += lognf
        self._weights[b1] = nw1
        self._tensors[0] = new_left

    @classmethod
    def load_from_file(cls, savefile, *, verbosity=0):
        """
        Load simple update from given file.

        Parameters
        ----------
        savefile : str
            Path to save file, as created by save_to_file.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        if verbosity > 0:
            print("Restart SymmetrizedSimpleUpdate back from file", savefile)
        with np.load(savefile) as fin:
            if fin["_SimpleUpdate_classname"] != cls._classname:
                msg = f"Savefile '{savefile}' does not match class '{cls._classname}'"
                raise ValueError(msg)
            D = fin["_SimpleUpdate_D"][()]
            beta = fin["_SimpleUpdate_beta"][()]
            tau = fin["_SimpleUpdate_tau"][()]
            logZ = fin["_SimpleUpdate_logZ"][()]
            rcutoff = fin["_SimpleUpdate_rcutoff"][()]
            degen_ratio = fin["_SimpleUpdate_degen_ratio"][()]

            ST = get_symmetric_tensor_type(fin["_SimpleUpdate_symmetry"][()])
            raw_hamilts = [
                ST.load_from_dic(fin, prefix=f"_SimpleUpdate_hamiltonian_{i}")
                for i in range(fin["_SimpleUpdate_n_hamiltonians"])
            ]

            A = ST.load_from_dic(fin, prefix="_SimpleUpdate_tensor_0")

            weights = []
            for i in range(4):
                prefix = f"_SimpleUpdate_weights_{i}"
                w = DiagonalTensor.load_from_dic(fin, prefix=prefix)
                weights.append(w)

        return cls(
            D,
            beta,
            tau,
            rcutoff,
            degen_ratio,
            A,
            raw_hamilts,
            weights,
            logZ,
            verbosity=verbosity,
        )
