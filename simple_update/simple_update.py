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

    _unit_cell = NotImplemented
    _n_bonds = NotImplemented
    _n_hamiltonians = NotImplemented
    _n_tensors = NotImplemented

    def __init__(
        self,
        Dx,
        beta,
        tau,
        rcutoff,
        tensors,
        hamiltonians,
        weights,
        verbosity,
    ):
        """
        Initialize SimpleUpdate from values. Consider calling from_infinite_temperature
        or from_file class methods instead of calling directly __init__.

        Parameters
        ----------
        Dx : int
            Maximal number of independent multiplets to keep when truncating bonds. For
            abelian symmetries, this is the same as the bond dimension D.
        beta : float
            Inverse temperature.
        tau : float
            Imaginary time step.
        rcutoff : float
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        tensors : enumerable of SymmetricTensor
            List of tensors in the unit cell.
        hamiltonians : enumerable of SymmetricTensor
            List of Hamiltonians defined on the unit cell.
        weights : list of numpy array
            Simple update weights for each bond of the unit cell.
        verbosity : int
            Level of log verbosity.
        """

        if len(tensors) != self._n_tensors:
            raise ValueError("Invalid number of tensors")
        if len(hamiltonians) != self._n_hamiltonians:
            raise ValueError("Invalid number of Hamiltonians")
        if len(weights) != self._n_bonds:
            raise ValueError("Invalid number of weights")

        self.verbosity = verbosity
        self._symmetry = tensors[0].symmetry
        if self.verbosity > 0:
            print(
                f"Construct SimpleUpdate with {self._symmetry} symmetry.",
                f"beta = {beta:.6g}",
            )
            print(f"unit cell:\n{self._unit_cell}")

        self.Dx = Dx
        self._beta = beta
        self._tensors = list(tensors)
        self._hamilts = list(hamiltonians)
        self.tau = tau  # also set gates
        self.rcutoff = rcutoff
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

        if self.verbosity > 1:
            print(self)

    @classmethod
    def from_infinite_temperature(
        cls, Dx, tau, hamiltonians, rcutoff=1e-10, verbosity=0
    ):
        """
        Initialize simple update at beta = 0 product state.

        Parameters
        ----------
        Dx : int
            Maximal number of independent multiplets to keep when truncating bonds. For
            abelian symmetries, this is the same as the bond dimension D.
        tau : float
            Imaginary time step.
        hamilts : enumerable of (d**2, d**2) ndarray
            Bond Hamltionians. Must be real symmetric or hermitian.
        rcutoff : float, optional.
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
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
            if cls._unit_cell != fin["_SimpleUpdate_unit_cell"]:
                raise ValueError("Savefile is incompatible with class unit cell")
            Dx = fin["_SimpleUpdate_Dx"][()]
            beta = fin["_SimpleUpdate_beta"][()]
            tau = fin["_SimpleUpdate_tau"][()]
            rcutoff = fin["_SimpleUpdate_rcutoff"][()]

            ST = get_symmetric_tensor_type(fin["_SimpleUpdate_symmetry"][()])
            hamiltonians = [
                ST.load_from_dic(fin, prefix=f"_SimpleUpdate_h_{i}")
                for i in range(cls._n_hamiltonians)
            ]
            tensors = [
                ST.load_from_dic(fin, prefix=f"_SimpleUpdate_t_{i}")
                for i in range(cls._n_tensors)
            ]

            weights = [None] * cls._n_bonds
            bond_representations = [None] * cls._n_bonds
            for i in range(cls._n_bonds):
                bond_representations[i] = fin[f"_SimpleUpdate_bond_rep_{i}"]
                weights[i] = fin[f"_SimpleUpdate_weights_{i}"]

        return cls(
            Dx,
            beta,
            tau,
            rcutoff,
            tensors,
            hamiltonians,
            bond_representations,
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
            "_SimpleUpdate_unit_cell": self._unit_cell,
            "_SimpleUpdate_Dx": self.Dx,
            "_SimpleUpdate_beta": self._beta,
            "_SimpleUpdate_tau": self._tau,
            "_SimpleUpdate_rcutoff": self.rcutoff,
        }
        for i, h in enumerate(self._hamiltonians):
            data |= h.get_data_dic(prefix=f"_SimpleUpdate_hamiltonian_{i}")

        for i, t in enumerate(self._tensors):
            data |= t.get_data_dic(prefix=f"_SimpleUpdate_tensor_{i}")

        for i in range(self._n_bonds):
            data[f"_SimpleUpdate_weights_{i}"] = self._weights[i]

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
        self._gates = [(-tau * h).expm() for h in self._hamilts]
        self._sqrt_gates = [(-0.5 * tau * h).expm() for h in self._hamilts]
        self._squared_gates = [(-2 * tau * h).expm() for h in self._hamilts]
        self._dbeta = 4 * tau  # 2nd order Trotter Suzuki + rho is quadratic in psi

    @property
    def beta(self):
        return self._beta

    def get_bond_representations(self):
        raise NotImplementedError("Must be defined in derived class")

    def bond_entanglement_entropy(self):
        """
        Compute the entanglement entropy on every bonds as s_ent = -sum_i p_i log_p_i
        """
        s_ent = np.empty(self._n_bonds)
        bond_rep = self.get_bond_representations()
        for i, (w, rep) in enumerate(zip(self._weights, bond_rep)):
            s_ent[i] = -w * np.log(w) @ rep.get_multiplet_structure()
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
        raise NotImplementedError

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
        effL.diagonal_imul(svL, left=True)
        cstR, svR, effR = right.svd()  # auxR-effR=p,m
        effR.diagonal_imul(svR, left=True)

        # change tensor structure to contract mid
        effL = effL.permutate((0, 1), (2,))  # auxL,p=effL-m
        effL.diagonal_imul([1.0 / w for w in weights])
        effR = effR.permutate((2,), (0, 1))  # m-effR=auxR,p

        # construct matrix theta and apply gate
        theta = effL @ effR  # auxL,pL=theta=auxR,pR
        theta = theta.permutate((0, 2), (1, 3))  # auxL, auxR = theta = pL, pR
        theta = theta @ gate

        # transpose back LxR, compute SVD and truncate
        theta = theta.permutate((0, 2), (1, 3))  # auxL, pL = theta = auxR, pR
        effL, new_weights, effR = theta.truncated_svd(self.Dx, rcutoff=self.rcutoff)

        # normalize weights and apply them to new left and new right
        new_weights = self._normalized_weights(new_weights, effL.col_reps[0])
        effL.diagonal_imul(new_weights)
        effR.diagonal_imul(new_weights, left=True)

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

            left            mid             right             gate
            /  \           /   \            /   \            /    \
           /    \         /\    \          /     \          /      \
         ///    /\       /  \   \\\      ///     /\        /\      /\
        auxL   pL mL    mL  mR  auxm    auxR    pR mR     pL pR    pL pR

        auxL, auxm and auxR can be anything, with any number of leg inside. They will be
        cut and stay inside constant parts, unaffected by the gate. pL and pR are the
        left and right physical variables. mL and mR are the left and right virtual
        bonds to be updated.
        """
        # 1) SVD cut between constant tensors and effective tensors to update
        cstL, svL, effL, auxL = left.svd()  # auxL - effL = pL, mL
        effL.diagonal_imul(svL, left=True)
        effL = effL.permutate((0, 1), (2,))  # auxL,pL = effL - mL
        effL.diagonal_imul([1.0 / w for w in weightsL])

        effm, svm, cstm, aux_m = mid.svd()  # mL, mR = effm - auxm
        effm.diagonal_imul(svm)
        effm = effm.permutate((0,), (1, 2))  # mL - effm = mR, auxm

        cstR, svR, effR, auxR = right.svd()  # auxR - effR = pR, mR
        effR.diagonal_imul(svR, left=True)
        effR = effR.permutate((2,), (1, 0))  # mR - effR = pR, auxR
        effR.diagonal_imul([1 / w for w in weightsR], left=True)

        # contract tensor network
        #                         ||
        #    =effL-weightsL -- effmid -- weightsR-effR=
        #         \                             /
        #          \----------- gate ----------/
        theta = effL @ effm  # auxL, pL = theta = mR, auxm
        theta = theta.permutate((0, 1, 3), (2,))  # auxL, pL, auxm = theta - mR
        theta = theta @ effR  # auxL, pL, auxm = theta = pR, auxR
        theta = theta.permutate((0, 2, 4), (1, 3))  # auxL, auxm, auxR = theta = pL, pR
        theta = theta @ gate

        # 1st SVD
        theta = theta.permutate((0, 3, 1), (4, 2))  # auxL, pL, auxm = theta = pR, auxR
        theta, new_weightsR, effR = theta.truncated_svd(self.Dx, rcutoff=self.rcutoff)
        new_weightsR = self._normalized_weights(new_weightsR, effR.row_reps[0])
        effR.diagonal_imul(new_weightsR, left=True)  # mR - effR = pR, auxR

        # 2nd SVD
        theta.diagonal_imul(new_weightsR)  # auxL, pL, auxm = theta = mR
        theta = theta.permutate((0, 1), (2, 3))  # auxL, pL = theta = auxm, mR
        effL, new_weightsL, effm = theta.truncated_svd(self.Dx, rcutoff=self.rcutoff)
        new_weightsL = self._normalized_weights(new_weightsL, effL.row_reps[0])
        effm.diagonal_imul(new_weightsL, left=True)  # mL - effm = auxm, mR
        effL.diagonal_imul(new_weightsL)  # auxL, pL = effL - mL

        # reshape to initial tree structure
        effL = effL.permutate((0,), (1, 2))  # auxL - effL = pL, mL
        effm = effm.permutate((0, 2), (1,))  # mL, mR = effm - auxm
        effR = effR.permutate((2,), (1, 0))  # auxR - effR = pR, mR

        # reconnect with const parts
        newL = cstL @ effL
        new_mid = effm @ cstm
        newR = cstR @ effR
        return newL, new_mid, newR, new_weightsL, new_weightsR
