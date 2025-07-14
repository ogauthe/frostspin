import numpy as np

from frostspin import DiagonalTensor
from frostspin.symmetric_tensor.tools import get_symmetric_tensor_type

from .abstract_simple_update import AbstractSimpleUpdate
from .simple_update_tools import check_hamiltonians, check_tensor_bond_indices
from .su_models import j1_j2_models


def decode_raw_update_data(raw_update_data, *, second_order=True):
    """
    Parameters
    ----------
    raw_update_data : int ndarray of shape (n, 6)
        Elementary information for each update. See notes for precise data format.
    second_order : bool
        Whether to generate second order update by reversing elementary order sequence.
        If False, output list length m is equal to n, else it is doubled m = 2*n - 2.

    Returns
    -------
    bond1 : list of size m
        Index of first updated bond in elementary update.
    bond2 : list of size m
        Index of second updated bonds in elementary update. Same as bond1 for a first
        neighbor elementary update.
    gate_indices : list of size m
        Index of gate in elementary update.
    left_indices : list of size m
        Index of left tensor in elementary update.
    right_indices : list of size m
        Index of right tensor in elementary update.
    middle_indices : list of size m
        Index of middle tensor in elementary update.

    Notes
    -----
    raw_update_data is a convenient way to store all information concerning updates in
    a compact form that can be easily stored in a npz file. Its format may evolve, the
    point is it is only read here to construct lists that are actually used in
    elementary updates.

    raw_update_data is a (n, 6) int array, where n is the number of first order updates.
    One row gives (bond1, bond2, gate_index, left_index, right_index, middle_index) for
    one elementary update. If bond1 == bond2, the update is first neighbor and the last
    column element middle_index is not read.
    """
    # also possible not to specify iL, iR and im and to recover them from b1 and b2
    # using tensor_bond_indices information. However this requires to set some
    # convention on which tensor is left or right, and this prevents fine control on
    # tensor signatures.
    update_data = np.asarray(raw_update_data, dtype=int)
    if second_order:
        update_data = np.vstack((update_data, update_data[-2:0:-1]))
    bond1 = list(update_data[:, 0])
    bond2 = list(update_data[:, 1])
    gate_indices = list(update_data[:, 2])
    left_indices = list(update_data[:, 3])
    right_indices = list(update_data[:, 4])
    middle_indices = [None] * update_data.shape[0]  # keep None for 1st neighbor
    for i, u in enumerate(update_data):
        if bond1[i] != bond2[i]:  # there is an intermediate site
            middle_indices[i] = u[5]
    return bond1, bond2, gate_indices, left_indices, right_indices, middle_indices


def get_update_data(
    tensor_bond_indices, raw_update_data, raw_hamilts, *, second_order=True
):
    """
    Construct all needed objects (swaps, indices...) used in evolve.
    """
    # this functions defines all variables needed to run updates. It is run only at
    # initialization, performances do not matter much here. The goal is to make
    # everything else simple and fast.

    # impose all tensors to have a physical and an ancilla leg as their two first
    # legs in default configuration, although they may be dummy legs.

    # define second order scheme
    bond1, bond2, gate_indices, left_indices, right_indices, middle_indices = (
        decode_raw_update_data(raw_update_data, second_order=second_order)
    )
    n_updates = len(bond1)
    n_tensors = len(tensor_bond_indices)
    hamiltonians = list(raw_hamilts)

    # find leg state after an update cycle
    eff_leg_state = [None] * n_tensors  # index of active legs after update
    for i in range(n_updates):
        eff_leg_state[left_indices[i]] = [-1, bond1[i]]
        eff_leg_state[right_indices[i]] = [-1, bond2[i]]
        if bond1[i] != bond2[i]:  # there is an intermediate site
            eff_leg_state[middle_indices[i]] = [bond1[i], bond2[i]]

    # now, leg_state is at the end of an update cycle: use it to define initial swap
    initial_swap = [None] * n_tensors
    leg_state = [None] * n_tensors
    for i in range(n_tensors):
        init = [-1, -2, *tensor_bond_indices[i]]
        end = sorted(set(init) - set(eff_leg_state[i])) + eff_leg_state[i]
        swap = [init.index(i) for i in end]
        initial_swap[i] = (tuple(swap[:-2]), tuple(swap[-2:]))
        leg_state[i] = end

    # construct swaps starting from initial state
    lperm = [None] * n_updates
    rperm = [None] * n_updates
    mperm = [None] * n_updates
    for i in range(n_updates):
        # left tensor
        init = leg_state[left_indices[i]]
        end = [*sorted(set(init) - {-1, bond1[i]}), -1, bond1[i]]
        swap = [init.index(i) for i in end]
        leg_state[left_indices[i]] = end
        lperm[i] = (tuple(swap[:-2]), tuple(swap[-2:]))

        # right tensor
        init = leg_state[right_indices[i]]
        end = [*sorted(set(init) - {-1, bond2[i]}), -1, bond2[i]]
        swap = [init.index(i) for i in end]
        leg_state[right_indices[i]] = end
        rperm[i] = (tuple(swap[:-2]), tuple(swap[-2:]))

        # middle tensor
        if bond1[i] != bond2[i]:
            init = leg_state[middle_indices[i]]
            b12 = [bond1[i], bond2[i]]
            end = sorted(set(init) - set(b12)) + b12
            swap = [init.index(i) for i in end]
            leg_state[middle_indices[i]] = end
            mperm[i] = (tuple(swap[:-2]), tuple(swap[-2:]))

    # add a final step identical to step 0
    for x in [
        bond1,
        bond2,
        gate_indices,
        left_indices,
        right_indices,
        middle_indices,
        lperm,
        rperm,
        mperm,
    ]:
        x.append(x[0])

    if second_order:
        # special case: i = n_update, corresponds to i = 0 with dbeta = tau
        eff_leg_state[left_indices[0]] = [-1, bond1[0]]
        eff_leg_state[right_indices[0]] = [-1, bond2[0]]
        if bond1[0] != bond2[0]:
            eff_leg_state[middle_indices[0]] = [bond1[0], bond2[0]]

        # we need two additional Hamiltonians, with a factor 2, for the 2 extremities of
        # the second order string, corresponding to gates with 2*tau
        hamiltonians.append(2 * raw_hamilts[gate_indices[0]])
        gate_indices[0] = len(raw_hamilts)
        hamiltonians.append(2 * raw_hamilts[gate_indices[n_updates // 2]])
        gate_indices[n_updates // 2] = len(raw_hamilts) + 1

    # Construct final swap that sends back to default state
    final_swap = [None] * n_tensors
    for i in range(n_tensors):
        end = [-1, -2, *tensor_bond_indices[i]]
        init = sorted(set(end) - set(eff_leg_state[i])) + eff_leg_state[i]
        swap = [init.index(i) for i in end]
        final_swap[i] = (tuple(swap[:2]), tuple(swap[2:]))

    return (
        bond1,
        bond2,
        gate_indices,
        left_indices,
        right_indices,
        middle_indices,
        lperm,
        rperm,
        mperm,
        hamiltonians,
        initial_swap,
        final_swap,
    )


class SimpleUpdate(AbstractSimpleUpdate):
    """
    Simple update algorithm implemented using SymmetricTensor. Only deals with thermal
    ensembles, pure wavefunctions are not implemented.

    Base class to be used as parent for different setup depending on geometry and unit
    cell. Class variables _unit_cell, _n_bond, _n_hamilt and_n_tensors must be defined
    in subclasses.
    """

    _classname = "SimpleUpdate"  # used in save/load to check consistency
    # 1st order should be fine but has not been tested
    _is_second_order = True

    def __init__(
        self,
        D,
        beta,
        tau,
        rcutoff,
        degen_ratio,
        tensor_bond_indices,
        tensors,
        raw_update_data,
        raw_hamilts,
        weights,
        logZ,
        *,
        verbosity=0,
    ):
        """
        Initialize SimpleUpdate from values.

        Parameters
        ----------
        D : int
            Bond dimension to keep when renormalizing bonds. This is a target, the
        actual largest value Dmax may differ due to cutoff or degeneracies.
        beta : float
            Inverse temperature.
        tau : float
            Imaginary time step.
        rcutoff : float
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        degen_ratio : float
            Consider singular values degenerate if their quotient is above degen_ratio.
        tensor_bond_indices : list of enumerable of int
            Information on which bonds belongs to which tensor. See notes for exact
            format.
        tensors : enumerable of SymmetricTensor
            List of simple update tensors, following tensor_bond_indices order and
            conventions. See notes for details.
        raw_update_data : 2D array
            Raw data on updates, exact format is specified in decode_raw_update_data.
        raw_hamilts : enumerable of SymmetricTensor
            List of elementary bond Hamiltonians acting on the tensors.
        weights : list of DiagonalTensor
            Simple update weights for each bond.
        logZ : float
            Logarithm of the PEPS norm. Used to compute free energy.
        verbosity : int
            Level of log verbosity.

        Notes
        -----
        tensor_bond_indices contains all information on the topology of the graph. It
        is a list of enumerable of int, where element i corresponds to the list of
        virtual legs for SymmetricTensor i in tensors. These virtual legs must be
        labelled from 0 to n_bonds and are associated with the simple update weights
        provided in weights.
        A given bond has to be shared by exactly two different tensors. A given tensor
        cannot carry twice the same virtual leg. Any graph following these constraints
        is allowed.

        SymmetricTensors in tensors must have one additional physical leg as first leg
        and one additional ancilla leg as second leg. They may be dummy legs with
        trivial representation if the tensor is purely virtual (no physical variable) or
        if it represents a pure wavefunction (no ancilla variable). The other legs must
        correspond to the bonds specified by tensor_bond_indices.
        """
        # input validation
        D = int(D)
        beta = float(beta)
        tau = float(tau)
        rcutoff = float(rcutoff)
        degen_ratio = float(degen_ratio)
        tensor_bond_indices = [np.array(tbi, dtype=int) for tbi in tensor_bond_indices]
        tensors = list(tensors)
        raw_update_data = np.asarray(raw_update_data, dtype=int)
        raw_hamilts = list(raw_hamilts)
        weights = list(weights)
        logZ = float(logZ)
        verbosity = int(verbosity)

        # quick crash for very simple errors
        check_tensor_bond_indices(tensor_bond_indices)
        check_hamiltonians(raw_hamilts)
        if len(tensors) != len(tensor_bond_indices):
            raise ValueError("Invalid tensor number")

        self.verbosity = int(verbosity)

        # set update data
        (
            self._1st_updated_bond,
            self._2nd_updated_bond,
            self._gate_indices,
            self._left_indices,
            self._right_indices,
            self._middle_indices,
            self._lperm,
            self._rperm,
            self._mperm,
            self._hamiltonians,
            self._initial_swap,
            self._final_swap,
        ) = get_update_data(tensor_bond_indices, raw_update_data, raw_hamilts)

        self._beta = beta
        self._n_bonds = len(weights)
        self._n_updates = len(self._1st_updated_bond) - 1
        self._n_tensors = len(tensors)
        self._symmetry = raw_hamilts[0].symmetry()
        self._tensor_bond_indices = tensor_bond_indices
        self._tensors = tensors
        self._weights = list(weights)
        self.D = D
        self._logZ = logZ
        self.rcutoff = rcutoff
        self.degen_ratio = degen_ratio
        self.tau = tau  # also set gates

        # raw data won't be used any more but may be saved to file
        self._raw_update_data = raw_update_data
        self._raw_hamilts = raw_hamilts

        self.check_consistency()

        if self.verbosity > 0:
            print(f"Construct SimpleUpdate with {self._symmetry} symmetry")
            print(f"beta = {beta:.6g}")

            if self.verbosity > 1:
                print(self)

    def __repr__(self):
        s = f"SimpleUpdate with {self._symmetry} symmetry and D = {self.D}"
        return s + f" at beta = {self._beta:.10g}"

    @classmethod
    def square_lattice_first_neighbor(
        cls, h1, D, tau, *, rcutoff=1e-10, degen_ratio=1.0, verbosity=0
    ):
        """
        Initialize first neighbor model on the square lattice at infinite temperature.
        """
        hamilts = [h1]
        tensor_bond_indices = [[0, 1, 2, 3], [2, 3, 0, 1]]
        update_data = np.array(
            [
                [0, 0, 0, 0, 1, -1],
                [1, 1, 0, 0, 1, -1],
                [2, 2, 0, 0, 1, -1],
                [3, 3, 0, 0, 1, -1],
            ]
        )
        return cls.from_infinite_temperature(
            D,
            tau,
            rcutoff,
            degen_ratio,
            tensor_bond_indices,
            update_data,
            hamilts,
            verbosity=verbosity,
        )

    @classmethod
    def square_lattice_second_neighbor(
        cls,
        h1,
        h2,
        D,
        tau,
        *,
        model="J1-J2",
        rcutoff=1e-10,
        degen_ratio=1.0,
        verbosity=0,
    ):
        """
        Initialize neighbor model on the square lattice at infinite temperature.

        The simplest model is the J1-J2, other similar models are implemented in models.

        For each second neighbor gate, Hamiltonian h2 will be applied twice with two
        different positions for the intermediate site.
        """
        hamilts = [h1, h2 / 2]
        tensor_bond_indices = [
            [0, 1, 2, 3],
            [4, 3, 5, 1],
            [2, 6, 0, 7],
            [5, 7, 4, 6],
        ]

        if model not in j1_j2_models:
            raise KeyError(f"Unknown model: {model}")
        update_data = j1_j2_models[model]
        return cls.from_infinite_temperature(
            D,
            tau,
            rcutoff,
            degen_ratio,
            tensor_bond_indices,
            update_data,
            hamilts,
            verbosity=verbosity,
        )

    @classmethod
    def from_infinite_temperature(
        cls,
        D,
        tau,
        rcutoff,
        degen_ratio,
        tensor_bond_indices,
        raw_update_data,
        raw_hamilts,
        *,
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
        rcutoff : float
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        degen_ratio : float
            Consider singular values degenerate if their quotient is above degen_ratio.
        tensor_bond_indices : list of enumerable of int
            Information on which bonds belongs to which tensor. See SimpleUpdate for
            more information on format.
        raw_update_data : 2D array
            Raw data on updates, exact format is specified in decode_raw_update_data.
        raw_hamilts : enumerable of SymmetricTensor
            List of elementary bond Hamiltonians acting on the tensors.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        check_tensor_bond_indices(tensor_bond_indices)

        n_tensors = len(tensor_bond_indices)
        n_bonds = max(max(tbi) for tbi in tensor_bond_indices) + 1
        ST = type(raw_hamilts[0])

        # find local representation by looking at Hamiltonians acting on tensor
        # ancilla leg signature never changes as this leg is never updated
        # physical lef signature also never changes as gates do not modify signature
        # virtual leg signatures may change depending on tensor position in update.

        data = get_update_data(tensor_bond_indices, raw_update_data, raw_hamilts)
        (
            bond1,
            bond2,
            gate_indices,
            left_indices,
            right_indices,
            middle_indices,
            _,
            _,
            _,
            hamiltonians,
            _,
            _,
        ) = data

        # initialize tensors to infinite temperature product state
        # if tensor is purely virtual and no Hamiltonian acts on it, add dummy leg
        phys_reps = [ST.singlet() for i in range(n_tensors)]
        signatures = [[None] * (2 + len(tbi)) for tbi in tensor_bond_indices]
        for j, b1 in enumerate(bond1):
            b2 = bond2[j]
            h = hamiltonians[gate_indices[j]]

            iL = left_indices[j]
            phys_reps[iL] = h.row_reps[0]
            signatures[iL][0] = ~h.signature[0]
            signatures[iL][1] = h.signature[0]
            leg = list(tensor_bond_indices[iL]).index(b1)
            signatures[iL][2 + leg] = True

            iR = right_indices[j]
            phys_reps[iR] = h.row_reps[1]
            signatures[iR][0] = ~h.signature[1]
            signatures[iR][1] = h.signature[1]
            leg = list(tensor_bond_indices[iR]).index(b2)
            signatures[iR][2 + leg] = False

            if b2 != b1:
                im = middle_indices[j]
                leg = list(tensor_bond_indices[im]).index(b1)
                signatures[im][2 + leg] = False
                leg = list(tensor_bond_indices[im]).index(b2)
                signatures[im][2 + leg] = True

        # initalize tensors
        tensors = []
        for i in range(n_tensors):
            d = ST.representation_dimension(phys_reps[i])
            sh = (d, d) + (1,) * len(tensor_bond_indices[i])
            mat = np.eye(d).reshape(sh)
            row_reps = (phys_reps[i], phys_reps[i])
            col_reps = (ST.singlet(),) * len(tensor_bond_indices[i])
            t = ST.from_array(mat, row_reps, col_reps, signature=signatures[i])
            tensors.append(t)

        beta = 0.0
        irr = t.block_irreps
        w = [np.ones((1,))]
        weights = [
            DiagonalTensor(w, ST.singlet(), irr, [1], ST.symmetry())
            for i in range(n_bonds)
        ]
        logZ = 0.0

        return cls(
            D,
            beta,
            tau,
            rcutoff,
            degen_ratio,
            tensor_bond_indices,
            tensors,
            raw_update_data,
            raw_hamilts,
            weights,
            logZ,
            verbosity=verbosity,
        )

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
            print("Restart SimpleUpdate back from file", savefile)
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

            tensors = []
            tensor_bond_indices = []
            for i in range(fin["_SimpleUpdate_n_tensors"]):
                t = ST.load_from_dic(fin, prefix=f"_SimpleUpdate_tensor_{i}")
                tbi = fin[f"_SimpleUpdate_tensor_bond_indices_{i}"]
                tensors.append(t)
                tensor_bond_indices.append(tbi)

            raw_update_data = fin["_SimpleUpdate_raw_update_data"]

            weights = []
            for i in range(fin["_SimpleUpdate_n_bonds"]):
                prefix = f"_SimpleUpdate_weights_{i}"
                w = DiagonalTensor.load_from_dic(fin, prefix=prefix)
                weights.append(w)

        return cls(
            D,
            beta,
            tau,
            rcutoff,
            degen_ratio,
            tensor_bond_indices,
            tensors,
            raw_update_data,
            raw_hamilts,
            weights,
            logZ,
            verbosity=verbosity,
        )

    def _elementary_update(self, i):
        """
        Elementary update bond between tensors iL and iR.
        """
        # This function is made to be fast: assume current state is fine and use
        # precomputed indices to select bonds, kind of update, tensors as well as
        # precomputed leg permutation.
        left = self._tensors[self._left_indices[i]].permute(*self._lperm[i])
        right = self._tensors[self._right_indices[i]].permute(*self._rperm[i])
        b1 = self._1st_updated_bond[i]
        b2 = self._2nd_updated_bond[i]
        if b1 == b2:  # 1st neighbor update
            left, right, nw1, lognf = self.update_first_neighbor(
                left, right, self._weights[b1], self._gates[self._gate_indices[i]]
            )

        else:  # update through middle site im
            mid = self._tensors[self._middle_indices[i]].permute(*self._mperm[i])
            left, mid, right, nw1, nw2, lognf = self.update_through_proxy(
                left,
                mid,
                right,
                self._weights[b1],
                self._weights[b2],
                self._gates[self._gate_indices[i]],
            )
            self._weights[b2] = nw2
            self._tensors[self._middle_indices[i]] = mid

        self._logZ += lognf
        self._weights[b1] = nw1
        self._tensors[self._left_indices[i]] = left
        self._tensors[self._right_indices[i]] = right

    def update_first_neighbor(self, left, right, weights, gate):
        r"""
        Update given bond by applying gate, computing the SVD and truncate the result.
        A 1D geometry is considered for clarity, the function being direction-agnostic.
        Left and right leg initial ordering is the same for simplicity.

        Parameters
        ----------
        left : SymmetricTensor
            "Left" tensor.
        right : SymmetricTensor
            "Right" tensor.
        weights : enumerable of numpy array
            Bond weights before update, as obtained by SymmetricTensor SVD
        gate : SymmetricTensor
            Gate to apply on the bond.

        Leg structures must be:

            left                right               gate
            /  \                /  \               /    \
           /    \              /    \             /      \
         ///    /\           ///    /\           /\      /\
        auxL   pL mid       auxR   pR mid      pL pR   pL pR

        auxL and auxR can be anything, with any number of leg inside. They will be cut
        and stay inside cstL and cstR, unaffected by the gate. pL and pR are the left
        and right physical variables. mid is the virtual bond to be updated.
        """
        # cut left and right between const and effective parts
        cstL, effL = left.qr()  # auxL-effL=p,m
        cstR, effR = right.qr()  # auxR-effR=p,m

        # change tensor structure to contract mid
        effL = effL.permute((0, 1), (2,))  # auxL,p=effL-m
        effL = effL * weights**-1
        effR = effR.permute((2,), (0, 1))  # m-effR=auxR,p

        # construct matrix theta and apply gate
        theta = effL @ effR  # auxL,pL=theta=auxR,pR
        theta = theta.permute((0, 2), (1, 3))  # auxL, auxR = theta = pL, pR
        theta = theta @ gate

        # transpose back LxR, compute SVD and truncate
        theta = theta.permute((0, 2), (1, 3))  # auxL, pL = theta = auxR, pR
        # define new_weights *on effL right*
        effL, new_weights, effR = theta.truncated_svd(
            self.D, rtol=self.rcutoff, degen_ratio=self.degen_ratio
        )

        # normalize weights and apply them to new left and new right
        # save log of normalization factor to update logZ
        new_weights /= new_weights.sum()
        lognf = np.log(theta.norm() / new_weights.norm())
        effL = effL * new_weights
        effR = new_weights * effR

        # reshape to initial tree structure
        effL = effL.permute((0,), (1, 2))  # auxL - effL = pL,m
        effR = effR.permute((1,), (2, 0))  # auxR - effR = pR,m

        # reconnect with const parts
        newL = cstL @ effL
        newR = cstR @ effR

        return newL, newR, new_weights, lognf

    def update_through_proxy(self, left, mid, right, weightsL, weightsR, gate):
        r"""
        Apply gate between two tensors through a proxy (either 2nd ord 3rd neighbor)
        A 1D geometry is considered for clarity, the function being direction-agnostic.

        Parameters
        ----------
        left : SymmetricTensor
            "Left" tensor
        mid : SymmetricTensor
            "Middle" tensor
        right : SymmetricTensor
            "Right" tensor
        weightsL : enumerable of numpy array
            Left bond weights before update.
        weightsR : enumerable of numpy array
            Right bond weights before update.
        gate : SymmetricTensor
            Gate to apply on the bond.

        Leg structures must be:

            left             mid             right             gate
            /  \            /   \            /   \            /    \
           /    \          /     \          /     \          /      \
         ///    /\       ///     /\       ///     /\        /\      /\
        auxL   pL mL    auxm   mL mR     auxR    pR mR     pL pR    pL pR

        auxL, auxm and auxR can be anything, with any number of leg inside. They will be
        cut and stay inside constant parts, unaffected by the gate. pL and pR are the
        left and right physical variables. mL and mR are the left and right virtual
        bonds to be updated.
        """
        # one must be careful with group representation conventions for weights. SVD of
        # M and M.T yields the same singular values, but not the same sorting due to
        # singular value irreps being conjugate. Always define new weights with singular
        # values *on the right* of effL and effR, and apply 1/weights *on the right*.

        # 1) SVD cut between constant tensors and effective tensors to update
        cstL, effL = left.qr()  # auxL - effL = pL, mL
        effL = effL.permute((0, 1), (2,))  # auxL,pL = effL - mL
        effL = effL * weightsL**-1

        cstm, effm = mid.qr()  # auxm - effm = mL, mR
        effm = effm.permute((1,), (2, 0))  # mL - effm = mR, auxm

        cstR, effR = right.qr()  # auxR - effR = pR, mR
        effR = effR.permute((0, 1), (2,))  # auR, pR = effR - mR
        effR = effR * weightsR**-1

        # contract tensor network
        #                         ||
        #    =effL-weightsL -- effmid -- weightsR-effR=
        #         \                             /
        #          \----------- gate ----------/
        theta = effL @ effm  # auxL, pL = theta = mR, auxm
        theta = theta.permute((0, 1, 3), (2,))  # auxL, pL, auxm = theta - mR
        theta = theta @ effR.transpose()  # auxL, pL, auxm = theta = auxR, pR
        theta = theta.permute((0, 2, 3), (1, 4))  # auxL, auxm, auxR = theta = pL, pR
        theta = theta @ gate

        # 1st SVD
        theta = theta.permute((4, 2), (0, 3, 1))  # pR, auxR = theta = auxL, pL, auxm
        norm0 = theta.norm()
        effR, new_weightsR, theta = theta.truncated_svd(
            self.D, rtol=self.rcutoff, degen_ratio=self.degen_ratio
        )
        new_weightsR /= new_weightsR.sum()
        effR = effR * new_weightsR  # pR, auxR = effR - mR

        # 2nd SVD
        theta = new_weightsR * theta  # mR-theta = auL,pL,auxm
        theta = theta.permute((1, 2), (3, 0))  # auxL, pL = theta = auxm, mR
        effL, new_weightsL, effm = theta.truncated_svd(
            self.D, rtol=self.rcutoff, degen_ratio=self.degen_ratio
        )
        new_weightsL /= new_weightsL.sum()
        lognf = np.log(norm0 / new_weightsL.norm())
        effm = new_weightsL * effm  # mL - effm = auxm, mR
        effL = effL * new_weightsL  # auxL, pL = effL - mL

        # reshape to initial tree structure
        effL = effL.permute((0,), (1, 2))  # auxL - effL = pL, mL
        effm = effm.permute((1,), (0, 2))  # auxm - effm = mL, mR
        effR = effR.permute((1,), (0, 2))  # auxR - effR = pR, mR

        # reconnect with const parts
        newL = cstL @ effL
        new_mid = cstm @ effm
        newR = cstR @ effR
        return newL, new_mid, newR, new_weightsL, new_weightsR, lognf

    def check_consistency(self):
        """
        Check whether a full cycle of updates can be executed without error.
        Does not care about how data is construted or save, does not read update_data.
        Only tensor_bond_indices and information called in evolve are used.
        """
        # check tensors match graph topology
        if len(self._tensors) != self._n_tensors:
            raise ValueError("Number of tensor does not match n_tensors")
        if len(self._tensor_bond_indices) != self._n_tensors:
            raise ValueError("Number of tensor_bond_indices does not match n_tensors")
        if max(max(tbi) for tbi in self._tensor_bond_indices) + 1 != self._n_bonds:
            raise ValueError("tensor_bond_indices does not match n_bonds")
        check_tensor_bond_indices(self._tensor_bond_indices)

        # Check all bonds are updated
        if len(self._weights) != self._n_bonds:
            raise ValueError("Number of weights does not match n_bonds")
        for b in range(self._n_bonds):
            if not (b in self._1st_updated_bond or b in self._2nd_updated_bond):
                raise ValueError(f"Bond {b} is never updated")
            if self._weights[b].symmetry() != self._symmetry:
                raise ValueError(f"Bond {b} has invalid symmetry")

        # check gates are 2-site operators
        check_hamiltonians(self._gates)

        # check update lists have correct length
        if len(self._1st_updated_bond) != self._n_updates + 1:
            raise ValueError("Invalid size for 1st_bond_indices")
        if len(self._2nd_updated_bond) != self._n_updates + 1:
            raise ValueError("Invalid size for 2nd_bond_indices")
        if len(self._gate_indices) != self._n_updates + 1:
            raise ValueError("Invalid size for gate_indices")
        if len(self._left_indices) != self._n_updates + 1:
            raise ValueError("Invalid size for left_indices")
        if len(self._right_indices) != self._n_updates + 1:
            raise ValueError("Invalid size for right_indices")
        if len(self._middle_indices) != self._n_updates + 1:
            raise ValueError("Invalid size for middle_indices")
        if len(self._lperm) != self._n_updates + 1:
            raise ValueError("Invalid size for lperm")
        if len(self._rperm) != self._n_updates + 1:
            raise ValueError("Invalid size for rperm")
        if len(self._mperm) != self._n_updates + 1:
            raise ValueError("Invalid size for mperm")

        if len(self._initial_swap) != self._n_tensors:
            raise ValueError("Invalid size for initial_swap")
        if len(self._final_swap) != self._n_tensors:
            raise ValueError("Invalid size for final_swap")

        def swap_legs(legs, swap):
            # swap has to match tensor legs and to produce a tensor with 2 column legs
            if len(swap) != 2 or len(swap[1]) != 2:
                raise ValueError("Invalid swap")
            sw = swap[0] + swap[1]
            if sorted(sw) != list(range(len(legs))):
                raise ValueError("Swap does not match legs")
            return [legs[i] for i in sw]

        def check_update(i, tensor_legs):
            b1 = self._1st_updated_bond[i]
            b2 = self._2nd_updated_bond[i]
            if not 0 <= b1 < self._n_bonds:
                raise ValueError(f"update {i}: invalid bond {b1}")
            if not 0 <= b2 < self._n_bonds:
                raise ValueError(f"update {i}: invalid bond {b2}")

            iL = self._left_indices[i]
            iR = self._right_indices[i]
            if not 0 <= iL < self._n_tensors:
                raise ValueError(f"update {i}: invalid tensor index {iL}")
            if not 0 <= iR < self._n_tensors:
                raise ValueError(f"update {i}: invalid tensor index {iR}")
            if iL == iR:
                raise ValueError(f"update {i}: left/right tensors are the same")

            ig = self._gate_indices[i]
            if not 0 <= ig < len(self._gates):
                raise ValueError(f"update {i}: invalid gate index {ig}")
            g = self._gates[ig]

            tL = self._tensors[iL]
            if tL.row_reps[0].shape != g.row_reps[0].shape:
                raise ValueError(f"update {i}: left/gate representations differ")
            if (tL.row_reps[0] != g.row_reps[0]).any():
                raise ValueError(f"update {i}: left/gate representations differ")
            if tL.signature[0] == g.signature[0]:
                raise ValueError(f"update {i}: left/gate signatures differ")

            tR = self._tensors[iR]
            if tR.row_reps[0].shape != g.row_reps[1].shape:
                raise ValueError(f"update {i}: right/gate representations differ")
            if (tR.row_reps[0] != g.row_reps[1]).any():
                raise ValueError(f"update {i}: right/gate representations differ")
            if tR.signature[0] == g.signature[1]:
                raise ValueError(f"update {i}: right/gate signature differ")

            tensor_legs[iL] = swap_legs(tensor_legs[iL], self._lperm[i])
            if tensor_legs[iL][-2] != -1:
                raise ValueError(f"update {i}: invalid left physical leg position")
            if tensor_legs[iL][-1] != b1:
                raise ValueError(f"update {i}: invalid left virtual leg position")
            tensor_legs[iR] = swap_legs(tensor_legs[iR], self._rperm[i])
            if tensor_legs[iR][-2] != -1:
                raise ValueError(f"update {i}: invalid right physical leg position")
            if tensor_legs[iR][-1] != b2:
                raise ValueError(f"update {i}: invalid right virtual leg position")

            rL = tL.col_reps[list(self._tensor_bond_indices[iL]).index(b1)]
            sL = tL.signature[2 + list(self._tensor_bond_indices[iL]).index(b1)]
            rR = tR.col_reps[list(self._tensor_bond_indices[iR]).index(b2)]
            sR = tR.signature[2 + list(self._tensor_bond_indices[iR]).index(b2)]
            if b1 == b2:  # 1st neighbor update
                if rL.shape != rR.shape or (rL != rR).any():
                    raise ValueError(f"update {i}: left/right legs do not match")
                if sL == sR:
                    raise ValueError(f"update {i}: left/right signatures do not match")

            else:  # update through middle site im
                im = self._middle_indices[i]
                if not 0 <= im < self._n_tensors:
                    raise ValueError(f"update {i}: invalid tensor index {im}")
                if iL == im:
                    raise ValueError(f"update {i}: left/middle tensors are the same")
                if im == iR:
                    raise ValueError(f"update {i}: right/middle tensors are the same")

                tensor_legs[im] = swap_legs(tensor_legs[im], self._mperm[i])
                if tensor_legs[im][-2] != b1:
                    raise ValueError(f"update {i}: invalid middle virtual leg position")
                if tensor_legs[im][-1] != b2:
                    raise ValueError(f"update {i}: invalid middle virtual leg position")
                tm = self._tensors[im]
                rm = tm.col_reps[list(self._tensor_bond_indices[im]).index(b1)]
                sm = tm.signature[2 + list(self._tensor_bond_indices[im]).index(b1)]
                if rL.shape != rm.shape or (rL != rm).any():
                    raise ValueError(f"update {i}: left and middle legs do not match")
                if sL == sm:
                    raise ValueError(f"update {i}: left/middle signatures do not match")
                rm = tm.col_reps[list(self._tensor_bond_indices[im]).index(b2)]
                sm = tm.signature[2 + list(self._tensor_bond_indices[im]).index(b2)]
                if rR.shape != rm.shape or (rR != rm).any():
                    raise ValueError(f"update {i}: right and middle legs do not match")
                if sR == sm:
                    raise ValueError(
                        f"update {i}: right/middle signatures do not match"
                    )

        tensor_legs = []
        ST = type(self._tensors[0])
        for i, t in enumerate(self._tensors):
            if type(t) is not ST:
                raise ValueError(f"Invalid type for tensor {i}")
            if t.ndim != 2 + len(self._tensor_bond_indices[i]):
                raise ValueError(f"Tensor {i} does not match tensor_bond_indices")
            if i not in self._left_indices + self._right_indices + self._middle_indices:
                raise ValueError(f"Tensor {i} is never updated")
            if t.n_row_reps != 2:
                raise ValueError(f"Tensor {i} has invalid leg structure")
            for j, bi in enumerate(self._tensor_bond_indices[i]):
                if t.col_reps[j].shape != self._weights[bi].representation.shape:
                    raise ValueError(f"Tensor {i} has invalid representation shape")
                if (t.col_reps[j] != self._weights[bi].representation).any():
                    raise ValueError(f"Tensor {i} has invalid representation")
            tl0 = [-1, -2, *self._tensor_bond_indices[i]]
            tl = swap_legs(tl0, self._initial_swap[i])
            tensor_legs.append(tl)

        # run full cycle of updates
        check_update(self._n_updates, tensor_legs)
        for i in range(1, self._n_updates):
            check_update(i, tensor_legs)

        # check state after init same as now
        for i in range(self._n_tensors):
            tl0 = [-1, -2, *self._tensor_bond_indices[i]]
            tl = swap_legs(tl0, self._initial_swap[i])
            if tl != tensor_legs[i]:
                raise ValueError(f"Tensor {i} does not come back to initial state")

        for x in [
            self._1st_updated_bond,
            self._2nd_updated_bond,
            self._left_indices,
            self._right_indices,
            self._middle_indices,
            self._lperm,
            self._rperm,
            self._mperm,
        ]:
            if x[0] != x[self._n_updates]:
                raise ValueError("Last and first update differ")

        if self._is_second_order:
            check_update(self._n_updates, tensor_legs)
            g0 = self._gates[self._gate_indices[0]]
            g1 = self._gates[self._gate_indices[self._n_updates]]
            if (g0 - g1 @ g1).norm() > 1e-14 * g1.norm():
                raise ValueError("Squared gate is not squared")
        elif self._gate_indices[0] != self._gates_indices[self._n_updates]:
            raise ValueError("Last and first update differ")

        for i in range(self._n_tensors):
            if len(self._final_swap[i]) != 2 or len(self._final_swap[i][0]) != 2:
                raise ValueError("Invalid final swap")
            sw = self._final_swap[i][0] + self._final_swap[i][1]
            if sorted(sw) != list(range(len(tensor_legs[i]))):
                raise ValueError("Swap does not match legs")
            tl = [tensor_legs[i][j] for j in sw]
            tl0 = [-1, -2, *self._tensor_bond_indices[i]]
            if tl != tl0:
                raise ValueError(f"Tensor {i} does not come back to default state")
