import numpy as np

from symmetric_tensor.tools import get_symmetric_tensor_type
from symmetric_tensor.non_abelian_symmetric_tensor import NonAbelianSymmetricTensor


class SimpleUpdate:
    """
    Simple update algorithm implemented using SymmetricTensor. Only deals with thermal
    ensembles, pure wavefunctions are not implemented.

    Base class to be used as parent for different setup depending on geometry and unit
    cell. Class variables _unit_cell, _n_bond, _n_hamilt and_n_tensors must be defined
    in subclasses.
    """

    _classname = "SimpleUpdate"  # used in save/load to check consistency

    def __init__(
        self,
        D,
        beta,
        tau,
        rcutoff,
        degen_ratio,
        tensors,
        tensor_bond_indices,
        update_data,
        hamiltonians,
        weights,
        verbosity,
    ):
        """
        Initialize SimpleUpdate from values. Consider calling from_infinite_temperature
        or from_file class methods instead of calling directly __init__.

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
        tensors : enumerable of SymmetricTensor
            List of tensors in the unit cell.
        hamiltonians : enumerable of SymmetricTensor
            List of Hamiltonians defined on the unit cell.
        weights : list of numpy array
            Simple update weights for each bond of the unit cell.
        verbosity : int
            Level of log verbosity.
        """
        self._n_tensors = len(tensors)

        # check consistency: virtual legs must be labelled 0 to n_bonds - 1
        # any virtual leg must appear exactly twice, and not twice on the same
        # tensor
        if len(tensor_bond_indices) != self._n_tensors:
            raise ValueError("Number of tensor does not match tensor_bond_indices")

        d = {}
        for i in range(self._n_tensors):
            tbi = list(tensor_bond_indices[i])
            for j, leg in enumerate(tbi):
                if tbi.index(leg) != j:
                    raise ValueError(f"Leg {leg} apperars twice in tensor {i}")
                try:
                    d[leg] += 1
                except KeyError:
                    d[leg] = 1
        if sorted(d.keys()) != list(range(len(weights))):
            raise ValueError("Bond indices must be 0 to n_bonds - 1")
        for bi in range(len(weights)):
            if d[bi] != 2:
                raise ValueError(f"Virtual bond {bi} does not appear exactly twice")

        self._n_hamiltonians = len(hamiltonians)
        self._n_bonds = len(weights)
        self._tensor_bond_indices = tensor_bond_indices
        self.verbosity = verbosity
        self._symmetry = tensors[0].symmetry
        self._symmetry = None
        if self.verbosity > 0:
            print(
                f"Construct SimpleUpdate with {self._symmetry} symmetry,",
                f"beta = {beta:.6g}",
            )

        self.D = D
        self._beta = beta
        self._tensors = list(tensors)
        self._raw_hamilts = list(hamiltonians)
        self.rcutoff = rcutoff
        self.degen_ratio = degen_ratio
        self._weights = weights

        # quick and dirty: define on the fly method to normalize weights
        self._ST = type(tensors[0])
        if issubclass(self._ST, NonAbelianSymmetricTensor):

            def normalized_weights(weights, rep):
                assert len(weights) == rep.shape[1]
                x = 0
                for i, w in enumerate(weights):
                    x += self._ST.irrep_dimension(rep[1, i]) * w.sum()
                return [w / x for w in weights]

        else:

            def normalized_weights(weights, rep):
                x = sum(w.sum() for w in weights)
                return [w / x for w in weights]

        self._normalized_weights = normalized_weights
        self.set_update_data(update_data)
        self.tau = tau  # also set gates

        if self.verbosity > 1:
            print(self)

    def __str__(self):
        s = repr(self)
        s = s + f"\nDmax = {self.Dmax}, tau = {self._tau}, rcutoff = {self.rcutoff}, "
        s = s + f"degen_ratio = {self.degen_ratio}"
        return s

    @classmethod
    def from_infinite_temperature(
        cls, D, tau, hamiltonians, rcutoff, degen_ratio, verbosity
    ):
        """
        Initialize simple update at beta = 0 product state.

        Parameters
        ----------
        D : int
            Maximal number of independent multiplets to keep when truncating bonds. For
            abelian symmetries, this is the same as the bond dimension D.
        tau : float
            Imaginary time step.
        hamilts : enumerable of (d**2, d**2) ndarray
            Bond Hamltionians. Must be real symmetric or hermitian.
        rcutoff : float, optional.
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        degen_ratio : float
            Consider singular values degenerate if their quotient is above degen_ratio.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        raise NotImplementedError

    @classmethod
    def load_from_file(cls, savefile, verbosity=0):
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
            rcutoff = fin["_SimpleUpdate_rcutoff"][()]
            degen_ratio = fin["_SimpleUpdate_degen_ratio"][()]

            ST = get_symmetric_tensor_type(fin["_SimpleUpdate_symmetry"][()])
            hamiltonians = [
                ST.load_from_dic(fin, prefix=f"_SimpleUpdate_hamiltonian_{i}")
                for i in range(fin["_SimpleUpdate_n_hamiltonians"])
            ]
            tensors = [
                ST.load_from_dic(fin, prefix=f"_SimpleUpdate_tensor_{i}")
                for i in range(fin["_SimpleUpdate_n_tensors"])
            ]
            tensor_bond_indices = fin["_SimpleUpdate_tensor_bond_indices"]
            update_data = fin["_SimpleUpdate_update_data"]

            # weights are list of numpy array, one list for one symmetry sector
            weights = [None] * cls._n_bonds
            for i in range(cls._n_bonds):
                n = fin[f"_SimpleUpdate_nw{i}"]
                weights[i] = [fin[f"_SimpleUpdate_weights_{i}_{j}"] for j in range(n)]

        return cls(
            D,
            beta,
            tau,
            rcutoff,
            degen_ratio,
            tensors,
            tensor_bond_indices,
            update_data,
            hamiltonians,
            weights,
            verbosity,
        )

    def save_to_file(self, savefile, additional_data={}):
        """
        Save SimpleUpdate in given file.

        Parameters
        ----------
        savefile : str
            Path to savefile.
        """
        data = {
            "_SimpleUpdate_symmetry": self._symmetry,
            "_SimpleUpdate_classname": self._classname,
            "_SimpleUpdate_D": self.D,
            "_SimpleUpdate_beta": self._beta,
            "_SimpleUpdate_tau": self._tau,
            "_SimpleUpdate_rcutoff": self.rcutoff,
            "_SimpleUpdate_degen_ratio": self.degen_ratio,
            "_SimpleUpdate_update_data": self._update_data,
            "_SimpleUpdate_tensor_bond_indices,": self._tensor_bond_indices,
            "_SimpleUpdate_n_tensors": self._n_tensors,
            "_SimpleUpdate_n_hamiltonians": self._n_hamiltonians,
        }
        for i, h in enumerate(self._hamilts):
            data |= h.get_data_dic(prefix=f"_SimpleUpdate_hamiltonian_{i}")

        for i, t in enumerate(self._tensors):
            data |= t.get_data_dic(prefix=f"_SimpleUpdate_tensor_{i}")

        for i in range(self._n_bonds):
            n = len(self._weights[i])
            data[f"_SimpleUpdate_nw{i}"] = n
            for j in range(n):
                data[f"_SimpleUpdate_weights_{i}_{j}"] = self._weights[i][j]

        np.savez_compressed(savefile, **data, **additional_data)
        if self.verbosity > 0:
            print("Simple update saved in file", savefile)

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self.verbosity > 0:
            print(f"set tau to {tau}")
        self._tau = tau
        self._gates = tuple((-tau * h).expm() for h in self._hamiltonians)
        self._dbeta = 4 * tau  # 2nd order Trotter Suzuki + rho is quadratic in psi

    @property
    def beta(self):
        return self._beta

    @property
    def Dmax(self):
        br = self.get_bond_representations()
        return max(self._ST.representation_dimension(r) for r in br)

    @property
    def symmetry(self):
        return self._ST.symmetry

    def get_bond_representation(self, i):
        for j in range(self._n_tensors):
            ind = (self._tensor_bond_indices[j] == i).nonzero()[0]
            if ind.size:
                return self._tensors[j].col_reps[ind[0]]
        raise ValueError(f"Unknown bond {i}")

    def get_bond_representations(self):
        """
        Obtain representations associated with all unit cell bonds
        """
        reps = []
        for i in range(self._n_bonds):
            # find tensor having this bond
            r = self.get_bond_representation(i)
            reps.append(r)
        return reps

    def bond_entanglement_entropy(self):
        """
        Compute the entanglement entropy on every bonds as s_ent = -sum_i p_i log_p_i
        """
        s_ent = np.empty(self._n_bonds)
        dw = self.get_weights()  # dense
        s_ent = np.array([-w @ np.log(w) for w in dw])
        return s_ent

    def get_weights(self, sort=True):
        """
        Return simple update weights for each bond with degeneracies.
        """
        if issubclass(self._ST, NonAbelianSymmetricTensor):
            # self._weights is a list of size self._n_bonds
            # self._weights[i] is a tuple of numpy arrays, as produced by ST.svd
            # self._weights[i][j] is a numpy array of shape (deg,), corresponding to
            # irrep irr, where deg and irr are given by get_bond_representations[i].
            weights = []
            for (wt, rep) in zip(self._weights, self.get_bond_representations()):
                dw = np.empty(self._ST.representation_dimension(rep))
                k = 0
                for (w, deg, irr) in zip(wt, rep[0], rep[1]):
                    dim = self._ST.irrep_dimension(irr)
                    for i in range(deg):
                        dw[k : k + dim] = w[i]
                        k += dim
                weights.append(dw)
        else:
            weights = [np.concatenate(w) for w in self._weights]
        if sort:
            for i, w in enumerate(weights):
                w.sort()
                weights[i] = w[::-1]
        return weights

    def get_tensors(self):
        """
        Returns
        -------
        tensors : tuple of _n_tensors SymmetricTensor
            Optimized tensors, with sqrt(weights) on all virtual legs.
        """
        # adding 1/sqrt(weights) is simpler in dense form
        sqw = [
            [1.0 / np.sqrt(w) for w in self.get_weights(sort=False)]
            for i in range(self._nbonds)
        ]
        tensors = []
        for i in range(self._n_tensors):
            # we already imposed the two first legs to be physical and ancilla in the
            # default configuration. Add weights on the virtual legs.
            t0 = self._tensors[i]
            args = [t0, list(range(t0.ndim))]
            for j, leg in enumerate(self._tensor_bond_indices[i]):
                args.append(sqw[leg])
                args.append([j])

            args.append(list(range(t0.ndim)))
            t = np.einsum(*args)
            t = self._ST.from_array(t, t0.row_reps, t0.col_reps, signature=t0.signature)
            tensors.append(t)
        return tensors

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
        cstL, svL, effL = left.svd()  # auxL-effL=p,m
        effL = effL.diagonal_mul(svL, left=True)
        cstR, svR, effR = right.svd()  # auxR-effR=p,m
        effR = effR.diagonal_mul(svR, left=True)

        # change tensor structure to contract mid
        effL = effL.permutate((0, 1), (2,))  # auxL,p=effL-m
        effL = effL.diagonal_mul([1.0 / w for w in weights])  # apply *on effL right*
        effR = effR.permutate((2,), (0, 1))  # m-effR=auxR,p

        # construct matrix theta and apply gate
        theta = effL @ effR  # auxL,pL=theta=auxR,pR
        theta = theta.permutate((0, 2), (1, 3))  # auxL, auxR = theta = pL, pR
        theta = theta @ gate

        # transpose back LxR, compute SVD and truncate
        theta = theta.permutate((0, 2), (1, 3))  # auxL, pL = theta = auxR, pR
        # define new_weights *on effL right*
        effL, new_weights, effR = theta.truncated_svd(
            self.D, rcutoff=self.rcutoff, degen_ratio=self.degen_ratio
        )

        # normalize weights and apply them to new left and new right
        new_weights = self._normalized_weights(new_weights, effL.col_reps[0])
        effL = effL.diagonal_mul(new_weights)
        effR = effR.diagonal_mul(new_weights, left=True)

        # reshape to initial tree structure
        effL = effL.permutate((0,), (1, 2))  # auxL - effL = pL,m
        effR = effR.permutate((1,), (2, 0))  # auxR - effR = pR,m

        # reconnect with const parts
        newL = cstL @ effL
        newR = cstR @ effR

        return newL, newR, new_weights

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
        cstL, svL, effL = left.svd()  # auxL - effL = pL, mL
        effL = effL.diagonal_mul(svL, left=True)
        effL = effL.permutate((0, 1), (2,))  # auxL,pL = effL - mL
        effL = effL.diagonal_mul([1.0 / w for w in weightsL])

        cstm, svm, effm = mid.svd()  # auxm - effm = mL, mR
        effm = effm.diagonal_mul(svm, left=True)
        effm = effm.permutate((1,), (2, 0))  # mL - effm = mR, auxm

        cstR, svR, effR = right.svd()  # auxR - effR = pR, mR
        effR = effR.diagonal_mul(svR, left=True)
        effR = effR.permutate((0, 1), (2,))  # auR, pR = effR - mR
        effR = effR.diagonal_mul([1.0 / w for w in weightsR])

        # contract tensor network
        #                         ||
        #    =effL-weightsL -- effmid -- weightsR-effR=
        #         \                             /
        #          \----------- gate ----------/
        theta = effL @ effm  # auxL, pL = theta = mR, auxm
        theta = theta.permutate((0, 1, 3), (2,))  # auxL, pL, auxm = theta - mR
        theta = theta @ effR.T  # auxL, pL, auxm = theta = auxR, pR
        theta = theta.permutate((0, 2, 3), (1, 4))  # auxL, auxm, auxR = theta = pL, pR
        theta = theta @ gate

        # 1st SVD
        theta = theta.permutate((4, 2), (0, 3, 1))  # pR, auxR = theta = auxL, pL, auxm
        effR, new_weightsR, theta = theta.truncated_svd(
            self.D, rcutoff=self.rcutoff, degen_ratio=self.degen_ratio
        )
        new_weightsR = self._normalized_weights(new_weightsR, effR.col_reps[0])
        effR = effR.diagonal_mul(new_weightsR)  # pR, auxR = effR - mR

        # 2nd SVD
        theta = theta.diagonal_mul(new_weightsR, left=True)  # mR-theta = auL,pL,auxm
        theta = theta.permutate((1, 2), (3, 0))  # auxL, pL = theta = auxm, mR
        effL, new_weightsL, effm = theta.truncated_svd(
            self.D, rcutoff=self.rcutoff, degen_ratio=self.degen_ratio
        )
        new_weightsL = self._normalized_weights(new_weightsL, effL.col_reps[0])
        effm = effm.diagonal_mul(new_weightsL, left=True)  # mL - effm = auxm, mR
        effL = effL.diagonal_mul(new_weightsL)  # auxL, pL = effL - mL

        # reshape to initial tree structure
        effL = effL.permutate((0,), (1, 2))  # auxL - effL = pL, mL
        effm = effm.permutate((1,), (0, 2))  # auxm - effm = mL, mR
        effR = effR.permutate((1,), (0, 2))  # auxR - effR = pR, mR

        # reconnect with const parts
        newL = cstL @ effL
        new_mid = cstm @ effm
        newR = cstR @ effR
        return newL, new_mid, newR, new_weightsL, new_weightsR

    def evolve(self, beta_evolve):
        """
        Evolve in imaginary time using second order Trotter-Suzuki up to beta.
        Convention: temperature value is the bilayer tensor one, twice the monolayer
        one.
        """
        if self.verbosity > 0:
            print(
                f"Evolve in imaginary time for beta from {self._beta:.6g} to "
                f"{self._beta + beta_evolve:.6g}..."
            )
        if beta_evolve < -1e-12:
            raise ValueError("Cannot evolve for negative imaginary time")
        if beta_evolve < 0.9 * self._dbeta:  # care for float rounding
            return  # else evolve for 1 step out of niter loop
        niter = round(beta_evolve / self._dbeta)  # 2nd order: evolve 2*tau by step

        # 2nd order update: update 0 is special as it may be squared or not
        # depending whether it is at the very beginning / end of update sequence
        self._initialize_update()
        for i in range(niter - 1):  # there is 1 step out of the loop
            for j in range(self._n_updates):
                self._elementary_update(j)
        self._finalize_update()
        self._beta += niter * self._dbeta

    def _elementary_update(self, i):
        """
        Elementary update bond between tensors iL and iR.
        """
        # This function is made to be fast: assume current state is fine and use
        # precomputed indices to select bonds, kind of update, tensors as well as
        # precomputed leg permutation.
        left = self._tensors[self._left_indices[i]].permutate(*self._lperm[i])
        right = self._tensors[self._right_indices[i]].permutate(*self._rperm[i])
        b1 = self._1st_updated_bond[i]
        b2 = self._2nd_updated_bond[i]
        if b1 == b2:  # 1st neighbor update
            left, right, nw1 = self.update_first_neighbor(
                left, right, self._weights[b1], self._gates[self._gate_indices[i]]
            )

        else:  # update through middle site im
            mid = self._tensors[self._middle_indices[i]].permutate(*self._mperm[i])
            left, mid, right, nw1, nw2 = self.update_through_proxy(
                left,
                mid,
                right,
                self._weights[b1],
                self._weights[b2],
                self._gates[self._gate_indices[i]],
            )
            self._weights[b2] = nw2
            self._tensors[self._middle_indices[i]] = mid

        self._weights[b1] = nw1
        self._tensors[self._left_indices[i]] = left
        self._tensors[self._right_indices[i]] = right

    def _initialize_update(self):
        # For a given tensor, its structure is defined by tensor_bond_indices between
        # updates. This structure needs to change at each update the tensor is involved
        # in. To avoid swappping back and forth for each elementary update, the tensor
        # structure constraint is relaxed inside evolve and changes within updates, as
        # defined by lperm, rperm and mperm. One cycle of updates will leave tensors
        # in a given final state structure, one needs to start from this state to be
        # able to loop over it.
        for i in range(self._n_tensors):
            self._tensors[i] = self._tensors[i].permutate(*self._initial_swap[i])

        # run last elementary update with time step tau instead of 2*tau
        self._elementary_update(self._n_updates)

        # run one cycle of update but the first one
        for j in range(1, self._n_updates):
            self._elementary_update(j)

    def _finalize_update(self):
        # run the last one with time step tau
        self._elementary_update(self._n_updates)

        # come back to default form, as defined in tensor_bond_indices
        for i in range(self._n_tensors):
            self._tensors[i] = self._tensors[i].permutate(*self._final_swap[i])

    def set_update_data(self, update_data):
        """
        update_data is a convenient way to store all information concerning update in a
        compact form. Construct all needed objects (swaps, indices...) used in evolve.

        From update_data, automatically generate second order Trotter-Suzuki scheme.
        """
        # this functions defines all variables needed to run updates. It is run only at
        # initialization, performances do not matter much here. The goal is to make
        # everything else simple and fast.

        # impose all tensors to have a physical and an ancilla leg as their two first
        # legs in default configuration, although they may be dummy legs.

        # define second order scheme
        n_updates = 2 * update_data.shape[0] - 2
        update_data_2nd = np.vstack((update_data, update_data[-2:0:-1]))
        assert update_data_2nd.shape == (n_updates, 6)

        def check(bi, ti):
            if bi not in self._tensor_bond_indices[ti]:
                raise ValueError(f"Bond {bi} does not exist in tensor {ti}")

        eff_leg_state = [None] * self._n_tensors  # index of active legs after update
        bond1 = list(update_data_2nd[:, 0])
        bond2 = list(update_data_2nd[:, 1])
        gate_indices = list(update_data_2nd[:, 2])
        left_indices = list(update_data_2nd[:, 3])
        right_indices = list(update_data_2nd[:, 4])
        middle_indices = [None] * self._n_tensors
        for i, u in enumerate(update_data_2nd):
            check(bond1[i], left_indices[i])
            check(bond2[i], right_indices[i])
            eff_leg_state[left_indices[i]] = [-1, bond1[i]]
            eff_leg_state[right_indices[i]] = [-1, bond2[i]]
            if bond1[i] != bond2[i]:  # there is an intermediate site
                middle_indices[i] = u[5]
                check(bond1[i], middle_indices[i])
                check(bond2[i], middle_indices[i])
                eff_leg_state[middle_indices[i]] = [bond1[i], bond2[i]]

        # now, leg_state is at the end of an update cycle: use it to define initial swap
        initial_swap = [None] * self._n_tensors
        leg_state = [None] * self._n_tensors
        for i in range(self._n_tensors):
            init = [-1, -2] + list(self._tensor_bond_indices[i])
            end = sorted(set(init) - set(eff_leg_state[i])) + eff_leg_state[i]
            swap = [init.index(i) for i in end]
            initial_swap[i] = (tuple(swap[:-2]), tuple(swap[-2:]))
            leg_state[i] = end

        # check all tensors and all bonds are updated
        if None in eff_leg_state:
            raise ValueError(f"Tensor {leg_state.index(None)} is never updated")
        for b in range(self._n_bonds):
            if b not in bond1 and b not in bond2:
                raise ValueError(f"Bond {b} is never updated")

        # now that we know final and initial states, we can run again all updates to
        # compute swaps
        lperm = [None] * n_updates
        rperm = [None] * n_updates
        mperm = [None] * n_updates
        for i in range(n_updates):
            init = leg_state[left_indices[i]]
            end = sorted(set(init) - set([-1, bond1[i]])) + [-1, bond1[i]]
            swap = [init.index(i) for i in end]
            leg_state[left_indices[i]] = end
            lperm[i] = (tuple(swap[:-2]), tuple(swap[-2:]))

            init = leg_state[right_indices[i]]
            end = sorted(set(init) - set([-1, bond2[i]])) + [-1, bond2[i]]
            swap = [init.index(i) for i in end]
            leg_state[right_indices[i]] = end
            rperm[i] = (tuple(swap[:-2]), tuple(swap[-2:]))
            if bond1[i] != bond2[i]:
                init = leg_state[middle_indices[i]]
                b12 = [bond1[i], bond2[i]]
                end = sorted(set(init) - set(b12)) + b12
                swap = [init.index(i) for i in end]
                leg_state[middle_indices[i]] = end
                mperm[i] = (tuple(swap[:-2]), tuple(swap[-2:]))

        # special case: i = n_update, corresponds to i = 0 with dbeta = tau
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

        eff_leg_state[left_indices[0]] = [-1, bond1[0]]
        eff_leg_state[right_indices[0]] = [-1, bond2[0]]
        if bond1[0] != bond2[0]:
            eff_leg_state[middle_indices[0]] = [bond1[0], bond2[0]]

        # Define final swap, which contains addiditional update by first bond
        final_swap = [None] * self._n_tensors
        for i in range(self._n_tensors):
            end = [-1, -2] + list(self._tensor_bond_indices[i])
            init = sorted(set(end) - set(eff_leg_state[i])) + eff_leg_state[i]
            swap = [init.index(i) for i in end]
            final_swap[i] = (tuple(swap[:-2]), tuple(swap[-2:]))

        # we need two additional Hamiltonians, with a factor 2, for the 2 extremities of
        # the second order string, corresponding to gates with 2*tau
        hamiltonians = self._raw_hamilts.copy()
        hamiltonians.append(2 * self._raw_hamilts[gate_indices[0]])
        gate_indices[0] = self._n_hamiltonians
        hamiltonians.append(2 * self._raw_hamilts[gate_indices[n_updates // 2]])
        gate_indices[n_updates // 2] = self._n_hamiltonians + 1

        # set data
        self._update_data = update_data  # not used out of here but my be saved to file
        self._n_updates = n_updates
        self._left_indices = left_indices
        self._right_indices = right_indices
        self._middle_indices = middle_indices
        self._gate_indices = gate_indices
        self._lperm = lperm
        self._rperm = rperm
        self._mperm = mperm
        self._1st_updated_bond = bond1
        self._2nd_updated_bond = bond2
        self._initial_swap = initial_swap
        self._final_swap = final_swap
        self._hamiltonians = hamiltonians
