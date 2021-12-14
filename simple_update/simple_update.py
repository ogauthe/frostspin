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
        bond_representations,
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
        if len(bond_representations) != self._n_bonds:
            raise ValueError("Invalid number of representations")
        if len(weights) != self._n_bonds:
            raise ValueError("Invalid number of weights")

        self.verbosity = verbosity
        self._symmetry = tensors[0].symmetry
        self._d = hamiltonians[0].shape[0]
        self._a = self._d
        if self.verbosity > 0:
            print(
                f"Construct SimpleUpdate with {self._symmetry} symmetry.",
                f"d = {self._d}, D* = {Dx},",
                f"beta = {beta:.6g}",
            )
            print(f"unit cell:\n{self._unit_cell}")

        self.Dx = Dx
        self._beta = beta
        self._tensor_list = list(tensors)
        self._hamilt_list = list(hamiltonians)
        self.tau = tau  # also set gates
        self.rcutoff = rcutoff
        self._bond_representations = bond_representations
        self._weights = weights

        # quick and dirty: define on the fly method to normalize weights
        self._ST = type(tensors[0])
        if issubclass(self._ST, NonAbelianSymmetricTensor):

            def normalized_weights(weights, rep):
                assert len(weights) == len(rep)
                x = 0
                for i, w in enumerate(weights):
                    x += self._ST.irrep_dimension(rep[i][1]) * w.sum()
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
        cls, Dx, tau, hamiltonians, rcutoff=1e-11, verbosity=0
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
        h0 = hamiltonians[0]
        ST = type(h0)
        phys = h0.row_reps[0]
        d = h0.shape[0]
        t0 = np.eye(d).reshape(d, d, 1, 1, 1, 1)
        # singlet may not be irrep 0 (it is irrep 1 for SU(2)), workaround with hamilt
        if issubclass(ST, NonAbelianSymmetricTensor):
            sing = ST.init_representation(np.array([1]), np.array([1]))
        else:
            sing = ST.init_representation(np.array([1]), np.array([0]))
        t = ST.from_array(
            t0, (phys,), (phys, sing, sing, sing, sing), conjugate_columns=False
        )
        return cls(
            Dx,
            0.0,
            tau,
            rcutoff,
            [t] * cls._n_tensors,
            hamiltonians,
            [sing] * cls._n_bonds,
            [np.ones(1)] * cls._n_bonds,
            verbosity,
        )

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
            data[f"_SimpleUpdate_bond_rep_{i}"] = self._bond_representations[i]
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
        self._gates = [(-tau * h).expm() for h in self._hamilt_list]
        self._sqrt_gates = [(-0.5 * tau * h).expm() for h in self._hamilt_list]
        self._squared_gates = [(-2 * tau * h).expm() for h in self._hamilt_list]
        self._dbeta = 4 * tau  # 2nd order Trotter Suzuki + rho is quadratic in psi

    @property
    def d(self):
        return self._d

    @property
    def a(self):
        return self._a

    @property
    def beta(self):
        return self._beta

    def bond_entanglement_entropy(self):
        """
        Compute the entanglement entropy on every bonds as s_ent = -sum_i p_i log_p_i
        """
        s_ent = np.empty(self._n_bonds)
        for i, (w, rep) in enumerate(zip(self._weights, self._bond_representations)):
            s_ent[i] = -w * np.log(w) @ rep.get_multiplet_structure()
        return s_ent

    def get_weights(self, sort=True):
        """
        Return simple update weights for each bond with degeneracies.
        """
        if issubclass(self._ST, NonAbelianSymmetricTensor):
            weights = []
            for (w, rep) in zip(self._weights, self._bond_representations):
                dw = np.empty(self._ST.representation_dimension(rep))
                w_index = 0
                dw_index = 0
                for (deg, irr) in zip(rep[0], rep[1]):
                    dim = self._ST.irrep_dimension(irr)
                    for i in range(deg):
                        dw[dw_index : dw_index + dim] = w[w_index]
                        w_index += 1
                        dw_index += dim
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

        Parameters
        ----------
        matL0 : SU2_Matrix
            "Left" matrix, tree structure is defined below.
        matR0 : SU2_Matrix
            "Right" matrix, tree structure is defined below.
        weights : numpy array
            Bond weights before update.
        gate : SymmetricTensor
            Gate to apply on the bond.


            matL0              matR0
            /  \                /  \
           /    \              /    \
          /     /\            /\     \
        left   p mid        mid p    right
        """
        # cut L and R between const and effective parts
        cstL, svL, effL = left.svd()
        effL.diagonal_imul(svL, left=True)
        effR, svR, cstR = right.svd()
        effR.diagonal_imul(svR)

        # change tensor structure to contract mid
        effL = effL.permutate((0, 1), (2,))
        effL.diagonal_imul([1 / w for w in weights])
        effR = effR.permutate((0,), (1, 2))

        # construct matrix theta and apply gate
        theta = effL @ effR
        theta = theta.permutate((0, 3), (1, 2))
        theta = theta @ gate

        # transpose back LxR, compute SVD and truncate
        theta = theta.permutate((0, 2), (1, 3))
        effL, new_weights, effR = theta.truncated_svd(self.Dx, rcutoff=self.rcutoff)

        # normalize weights and apply them to new left and new right
        new_weights = self._normalized_weights(new_weights, effL.col_reps[0])
        effL.diagonal_imul(new_weights)
        effR.diagonal_imul(new_weights, left=True)

        # reshape to initial tree structure
        effL = effL.permutate((0,), (1, 2))
        effR = effR.permutate((0, 1), (2,))

        # reconnect with const parts
        newL = cstL @ effL
        newR = effR @ cstR

        return newL, newR, new_weights

    def update_through_proxy(self, left0, mid0, right0, weightsL, weightsR, gate):
        r"""
        Apply gate between two tensors through a proxy (either 2nd ord 3rd neighbor)
        A 1D geometry is considered for clarity, the function being direction-agnostic.

        Parameters
        ----------
        left0 : SymmetricTensor
            "Left" tensor
        mid0 : SymmetricTensor
            "Middle" tensor
        right0 : SymmetricTensor
            "Right" tensor
        weightsL : numpy array
            Left bond weights before update.
        weightsR : numpy array
            Right bond weights before update.
        gate : SymmetricTensor
            Gate to apply on the bond. Tree structure must be
            (self._d * self._d, self._d * self._d)


            matL0            mat_mid0         matR0
            /  \             /    \           /  \
           /    \           /\     \         /    \
          /     /\         /  \     \       /\     \
        auxL   p repL    repL repR auxm   repR p  auxR
        """
        # 1) SVD cut between constant tensors and effective tensors to update
        #     \|        \|
        #     -L-    -> -cstL==effL-lambda_L-
        #      |\        |       \
        cstL, svL, effL, auxL = left0.svd()
        effL.diagonal_imul(svL, left=True)
        effL = effL.permutate((0, 1), (2,))
        effL.diagonal_imul([1 / w for w in weightsL])

        #                       \|/|
        #                       cstM
        #     \|                 ||
        #     -M-   ->        --effM--
        #      |\
        eff_m, sv_m, cst_m, aux_m = mid0.svd()
        eff_m.diagonal_imul(sv_m)
        eff_m = eff_m.permutate((0,), (1, 2))

        #     \|                         \|
        #     -R-   ->    lambda_R-effR==cstR
        #      |\                         |\
        effR, svR, cstR, auxR = right0.svd()
        effR.diagonal_imul(svR, left=True)
        effR = effR.permutate((0,), (1, 2))
        effR.diagonal_imul([1 / w for w in weightsR], left=True)

        # contract tensor network
        #                         ||
        #    =effL-lambdaL -- eff_mid -- lambdaR-effR=
        #         \                             /
        #          \----------- gate ----------/
        theta = effL @ eff_m
        theta = theta.permutate((0, 1, 3), (2,))
        theta = theta @ effR
        theta = theta.permutate((0, 2, 4), (1, 3))
        theta = theta @ gate

        # 1st SVD
        #     theta
        #    /     \
        #  0,1,2   3, 4
        #  L M R   pL, pR
        theta = theta.permutate((0, 1, 3), (4, 2))
        theta, new_weightsR, effR = theta.truncated_svd(self.Dx, rcutoff=self.rcutoff)
        new_weightsR = self._normalized_weights(new_weightsR, effR.row_reps[0])
        effR.diagonal_imul(new_weightsR, left=True)

        # 2nd SVD
        theta.diagonal_imul(new_weightsR)
        theta = theta.permutate((0, 1), (2, 3))

        effL, new_weightsL, eff_m = theta.truncated_svd(self.Dx, rcutoff=self.rcutoff)
        new_weightsL = self._normalized_weights(new_weightsL, effL.row_reps[0])
        eff_m.diagonal_imul(new_weightsL, left=True)
        effL.diagonal_imul(new_weightsL)

        # reshape to initial tree structure
        effL = effL.permutate((0,), (1, 2))
        eff_m = eff_m.permutate((0, 1), (2,))
        effR = effR.permutate((0, 1), (2,))

        # reconnect with const parts
        newL = cstL @ effL
        new_mid = eff_m @ cst_m
        newR = effR @ cstR
        return newL, new_mid, newR, new_weightsL, new_weightsR
