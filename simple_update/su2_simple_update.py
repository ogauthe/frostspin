import numpy as np

from groups.su2_representation import (
    SU2_Representation,
    SU2_Matrix,
    get_projector_chained,
    construct_matrix_projector,
)


class SU2_SimpleUpdate(object):
    """
    SU(2) symmetric simple update algorithm. Only deals with finite temperature. Each
    site has to be an *irreducible* representation of SU(2).

    This class is not made for direct use, unit cell is not defined. Static variables
    _unit_cell, _n_bond, _n_hamilt and_n_tensors must be defined in subclasses.
    """

    def __init__(
        self,
        Dstar,
        beta,
        tau,
        rcutoff,
        hamilts_raw_data,
        bond_representations,
        weights,
        tensors_data,
        verbosity,
    ):
        """
        Initialize SU2_SimpleUpdate from values. Not meant for direct use, consider
        calling from_infinite_temperature or from_file class methods.

        Parameters
        ----------
        Dstar : int
            Number of independent SU(2) multiplets to keep when truncating bonds.
        beta : float
            Inverse temperature.
        tau : float
            Imaginary time step.
        rcutoff : float
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        hamilts_raw_data : list of numpy array with shape (d,)
            Raw data to construct SU2_Matrix for each Hamiltonian defined on the unit
            cell. Assume (d * d, d * d) SU(2) tree decomposition, where d is the local
            irrep.
        bond_representations : list of SU2_Representation
            SU(2) representation for each bond of the unit cell.
        weights : list of numpy array
            Simple update weights for each bond of the unit cell.
        tensors_data : list of numpy array
            Raw data for each SU(2) symmetric tensors of the unit cell. Default tree
            decomposition has to be defined in subclasses.
        verbosity : int
            Level of log verbosity.
        """

        if len(hamilts_raw_data) != self._n_hamilts:
            raise ValueError("Invalid number of Hamiltonians")
        if len(bond_representations) != self._n_bonds:
            raise ValueError("Invalid number of bond representations")
        if len(weights) != self._n_bonds:
            raise ValueError("Invalid number of weights")
        if len(tensors_data) != self._n_tensors:
            raise ValueError("Invalid number of tensors")

        self.verbosity = verbosity
        self._d = hamilts_raw_data[0].size
        if self.verbosity > 0:
            print(
                f"Construct SU2_SimpleUpdate with d = {self._d}, D* = {Dstar},",
                f"beta = {beta}",
            )
            print(f"unit cell:\n{self._unit_cell}")

        self.Dstar = Dstar
        self._beta = beta
        self._phys = SU2_Representation.irrep(self._d)
        self._a = self._d
        self._anc = self._phys
        d2 = self._phys * self._phys
        self._hamilts = [SU2_Matrix.from_raw_data(r, d2, d2) for r in hamilts_raw_data]
        self.tau = tau
        self.rcutoff = rcutoff

        self._bond_representations = bond_representations
        self._weights = weights
        self._tensors_data = tensors_data
        self.reset_isometries()
        if self.verbosity > 1:
            print(self)

    @classmethod
    def from_file(cls, file, verbosity=0):
        """
        Load simple update from given file.

        Parameters
        ----------
        file : str
            Save file containing data to restart computation from. Must follow
            save_to_file syntax.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        if verbosity > 0:
            print("Restart SU2_SimpleUpdate1x2 back from file", file)
        with np.load(file) as data:
            if cls._unit_cell != data["_SU2_SU_unit_cell"]:
                raise ValueError("Savefile is incompatible unit cell")
            Dstar = data["_SU2_SU_Dstar"][()]
            beta = data["_SU2_SU_beta"][()]
            tau = data["_SU2_SU_tau"][()]
            rcutoff = data["_SU2_SU_rcutoff"][()]

            hamilts_raw_data = [
                data[f"_SU2_SU_hamilts_raw_data_{i}"] for i in range(cls._n_hamilts)
            ]
            tensors_data = [
                data[f"_SU2_SU_tensors_data_{i}"] for i in range(cls._n_tensors)
            ]

            weights = [None] * cls._n_bonds
            bond_representations = [None] * cls._n_bonds
            for i in range(cls._n_bonds):
                degen = data[f"_SU2_SU_degen_bond_{i}"]
                irreps = data[f"_SU2_SU_irreps_bond_{i}"]
                bond_representations[i] = SU2_Representation(degen, irreps)
                weights[i] = data[f"_SU2_SU_weights_bond_{i}"]

        return cls(
            Dstar,
            beta,
            tau,
            rcutoff,
            hamilts_raw_data,
            bond_representations,
            weights,
            tensors_data,
            verbosity,
        )

    def save_to_file(self, file):
        """
        Save simple update in given file.

        Parameters
        ----------
        file : str
            Savefile.
        """
        data = {
            "_SU2_SU_unit_cell": self._unit_cell,
            "_SU2_SU_Dstar": self.Dstar,
            "_SU2_SU_beta": self._beta,
            "_SU2_SU_tau": self._tau,
            "_SU2_SU_rcutoff": self.rcutoff,
        }
        for i in range(self._n_hamilts):
            data[f"_SU2_SU_hamilts_raw_data_{i}"] = self._hamilts[i].to_raw_data()

        for i in range(self._n_tensors):
            data[f"_SU2_SU_tensors_data_{i}"] = self._tensors_data[i]

        for i in range(self._n_bonds):
            data[f"_SU2_SU_degen_bond_{i}"] = self._bond_representations[i].degen
            data[f"_SU2_SU_irreps_bond_{i}"] = self._bond_representations[i].irreps
            data[f"_SU2_SU_weights_bond_{i}"] = self._weights[i]

        np.savez_compressed(file, **data)
        if self.verbosity > 0:
            print("Simple update data stored in file", file)

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self.verbosity > 0:
            print(f"set tau to {tau}")
        self._tau = tau
        self._gates = [(-tau * h).expm() for h in self._hamilts]
        self._squared_gates = [(-2 * tau * h).expm() for h in self._hamilts]
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
        for i, w in enumerate(self._weights):
            s_ent[i] = -w * np.log(w) @ self._virt_rep_list[i].get_multiplet_structure()
        return s_ent

    def get_tensors(self):
        """
        Return optimized tensors.
        Tensors are obtained by adding relevant sqrt(lambda) to every leg of gammaX
        For each virtual axis, sort by decreasing weights (instead of SU(2) order)
        """
        # TODO
        raise NotImplementedError

    def update_first_neighbor(self, matL0, matR0, weights, virt_mid, gate):
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
        virt_mid : SU2_Representation
            Bond SU(2) representation before update.
        gate : SU2_Matrix
            Gate to apply on the bond. Tree structure must be
            (self._d * self._d, self._d * self._d)


            matL0              matR0
            /  \                /  \
           /    \              /    \
          /     /\            /\     \
        left   p mid        mid p    right
        """
        # cut L and R between const and effective parts
        cstL, svL, effL, virt_left = matL0.svd()
        effL = svL[:, None] * effL
        effR, svR, cstR, virt_right = matR0.svd()
        effR = effR * svR

        # change tensor structure to contract mid
        pL0 = construct_matrix_projector((virt_left,), (self._phys, virt_mid))
        pL1 = construct_matrix_projector((virt_left, self._phys), (virt_mid,))
        isoL1 = np.tensordot(pL1, pL0, ((0, 1, 2), (0, 1, 2)))
        vL1 = isoL1 @ effL.to_raw_data()
        matL1 = (
            SU2_Matrix.from_raw_data(vL1, virt_left * self._phys, virt_mid) / weights
        )

        pR0 = construct_matrix_projector((virt_mid, self._phys), (virt_right,))
        pR1 = construct_matrix_projector((virt_mid,), (self._phys, virt_right))
        isoR1 = np.tensordot(pR1, pR0, ((0, 1, 2), (0, 1, 2)))
        vR1 = isoR1 @ effR.to_raw_data()
        matR1 = SU2_Matrix.from_raw_data(vR1, virt_mid, self._phys * virt_right)

        # construct matrix theta and apply gate
        theta_mat = matL1 @ matR1
        theta = theta_mat.to_raw_data()
        pLR2 = construct_matrix_projector(
            (virt_left, self._phys), (self._phys, virt_right)
        )
        pLR3 = construct_matrix_projector(
            (virt_left, virt_right), (self._phys, self._phys)
        )
        iso_theta = np.tensordot(pLR3, pLR2, ((0, 2, 3, 1), (0, 1, 2, 3)))
        theta2 = iso_theta @ theta
        theta_mat2 = SU2_Matrix.from_raw_data(
            theta2, virt_left * virt_right, self._phys * self._phys
        )
        theta_mat3 = theta_mat2 @ gate

        # transpose back LxR, compute SVD and truncate
        theta3 = iso_theta.T @ theta_mat3.to_raw_data()
        theta_mat4 = SU2_Matrix.from_raw_data(
            theta3, virt_left * self._phys, self._phys * virt_right
        )
        U, new_weights, V, new_virt_mid = theta_mat4.svd(
            cut=self.Dstar, rcutoff=self.rcutoff
        )

        # normalize weights and apply them to new left and new right
        new_weights /= new_weights @ new_virt_mid.get_multiplet_structure()
        U1 = U * new_weights
        V1 = new_weights[:, None] * V

        # recompute reshape matrices only if needed
        if new_virt_mid != virt_mid:
            pL0 = construct_matrix_projector((virt_left,), (self._phys, new_virt_mid))
            pL1 = construct_matrix_projector((virt_left, self._phys), (new_virt_mid,))
            isoL1 = np.tensordot(pL1, pL0, ((0, 1, 2), (0, 1, 2)))

            pR0 = construct_matrix_projector((new_virt_mid, self._phys), (virt_right,))
            pR1 = construct_matrix_projector((new_virt_mid,), (self._phys, virt_right))
            isoR1 = np.tensordot(pR1, pR0, ((0, 1, 2), (0, 1, 2)))

        # reshape to initial tree structure
        new_vL = isoL1.T @ U1.to_raw_data()
        new_effL = SU2_Matrix.from_raw_data(
            new_vL, virt_left, self._phys * new_virt_mid
        )
        new_vR = isoR1.T @ V1.to_raw_data()
        new_effR = SU2_Matrix.from_raw_data(
            new_vR, new_virt_mid * self._phys, virt_right
        )

        # reconnect with const parts
        newL = cstL @ new_effL
        newR = new_effR @ cstR

        return newL, new_weights, newR, new_virt_mid


class SU2_SimpleUpdate1x2(SU2_SimpleUpdate):
    """
    SU(2) symmetric simple update algorithm on plaquette AB, with
    - 4 bonds
    - 1 Hamiltonian
    - 2 tensors

    Conventions for leg ordering
    ----------------------------
          1     3
          |     |
       4--A--2--B--4
          |     |
          3     1
    """

    _unit_cell = "AB"
    _n_bonds = 4
    _n_hamilts = 1
    _n_tensors = 2

    def __repr__(self):
        return f"SU2_SimpleUpdate1x2 for irrep {self._d}"

    def __str__(self):
        return (
            f"SU2_SimpleUpdate1x2 for irrep {self._d}\n"
            f"D* = {self.Dstar}\n"
            f"beta = {self._beta:.6g}\n"
            f"tau = {self._tau}\n"
            f"rcutoff = {self.rcutoff}\n"
            f"bond 1 representation: {self._bond_representations[0]}\n"
            f"bond 2 representation: {self._bond_representations[1]}\n"
            f"bond 3 representation: {self._bond_representations[2]}\n"
            f"bond 4 representation: {self._bond_representations[3]}"
        )

    @classmethod
    def from_infinite_temperature(cls, d, Dstar, tau, h, rcutoff=1e-11, verbosity=0):
        """
        Initialize simple update at beta = 0 product state.

        Parameters
        ----------
        d : int
            dimension of physical SU(2) irreducible reprsentation on each site.
        Dstar : int
            Maximal number of independent multiplets kept on each bond.
        tau : float
            Imaginary time step.
        h : (d**2, d**2) ndarray
            Hamltionian. Must be real symmetric or hermitian.
        rcutoff : float, optional.
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        if verbosity > 0:
            print("Initialize SU2_SimpleUpdate1x2 at beta = 0 thermal product state")

        if h.shape != (d ** 2, d ** 2):
            raise ValueError("invalid shape for Hamiltonian")

        phys = SU2_Representation.irrep(d)
        proj = construct_matrix_projector((phys, phys), (phys, phys), conj_right=True)
        proj = proj.reshape(d ** 4, d)
        h_raw_data = proj.T @ h.ravel()
        return cls(
            Dstar,
            0.0,
            tau,
            rcutoff,
            [h_raw_data],
            [SU2_Representation.irrep(1)] * 4,
            [np.ones(1)] * 4,
            [np.ones(1)] * 2,
            verbosity,
        )

    def evolve(self, beta_evolve=None):
        """
        Evolve in imaginary time using second order Trotter-Suzuki up to beta.
        Convention: temperature value is the bilayer tensor one, twice the monolayer
        one.
        """
        if beta_evolve is None:
            beta_evolve = self._dbeta
        if self.verbosity > 0:
            print(
                f"Evolve in imaginary time for beta from {self._beta:.6g} to "
                f"{self._beta + beta_evolve:.6g}..."
            )
        if beta_evolve < -1e-16:
            raise ValueError("Cannot evolve for negative imaginary time")
        if beta_evolve < 0.9 * self._dbeta:  # care for float rounding
            return  # else evolve for 1 step out of niter loop
        niter = round(beta_evolve / self._dbeta)  # 2nd order: evolve 2*tau by step

        self.update_bond(1, self._gates[0])
        for i in range(niter - 1):  # there is 1 step out of the loop
            self.update_bond(2, self._gates[0])
            self.update_bond(3, self._gates[0])
            self.update_bond(4, self._squared_gates[0])
            self.update_bond(3, self._gates[0])
            self.update_bond(2, self._gates[0])
            self.update_bond(1, self._squared_gates[0])
            self._beta += self._dbeta
        self.update_bond(2, self._gates[0])
        self.update_bond(3, self._gates[0])
        self.update_bond(4, self._squared_gates[0])
        self.update_bond(3, self._gates[0])
        self.update_bond(2, self._gates[0])
        self.update_bond(1, self._gates[0])
        self._beta += self._dbeta
        if self.verbosity > 0:
            print(f"Done, beta = {self._beta:.6g}")

    def reset_isometries(self):
        if self.verbosity > 0:
            print(f"reset isometries at beta = {self._beta:.6g}")
        self._isoA1 = None
        self._isoB1 = None
        self._isoA2 = None
        self._isoB2 = None
        self._isoA3 = None
        self._isoB3 = None
        self._isoA4 = None
        self._isoB4 = None

    def get_isoAB1(self):
        if self._isoA1 is None:
            if self.verbosity > 1:
                eff1 = self._phys * self._bond_representations[0]
                aux1 = (
                    self._anc
                    * self._bond_representations[1]
                    * self._bond_representations[2]
                    * self._bond_representations[3]
                )
                print(f"compute isoA1 and isoB1: eff_rep1 = {eff1}")
                print(f"aux_rep1 = {aux1}")
            p_data = get_projector_chained(
                self._phys, self._anc, *self._bond_representations, singlet_only=True
            )
            p_data = p_data.reshape(-1, p_data.shape[6])
            p_transpA = construct_matrix_projector(
                (
                    self._anc,
                    self._bond_representations[1],
                    self._bond_representations[2],
                    self._bond_representations[3],
                ),
                (self._phys, self._bond_representations[0]),
            )
            p_transpA = p_transpA.transpose(4, 0, 5, 1, 2, 3, 6).reshape(p_data.shape)
            self._isoA1 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._bond_representations[0], self._phys),
                (
                    self._anc,
                    self._bond_representations[1],
                    self._bond_representations[2],
                    self._bond_representations[3],
                ),
            )
            p_transpB = p_transpB.transpose(1, 2, 0, 3, 4, 5, 6).reshape(p_data.shape)
            self._isoB1 = p_transpB.T @ p_data
        return self._isoA1, self._isoB1

    def get_isoAB2(self):
        if self._isoA2 is None:
            if self.verbosity > 1:
                eff2 = self._phys * self._bond_representations[1]
                aux2 = (
                    self._anc
                    * self._bond_representations[0]
                    * self._bond_representations[2]
                    * self._bond_representations[3]
                )
                print(f"compute isoA2 and isoB2: eff_rep2 = {eff2}")
                print(f"aux_rep2 = {aux2}")
            p_data = get_projector_chained(
                self._phys, self._anc, *self._bond_representations, singlet_only=True
            )
            p_data = p_data.reshape(-1, p_data.shape[6])
            p_transpA = construct_matrix_projector(
                (
                    self._anc,
                    self._bond_representations[0],
                    self._bond_representations[2],
                    self._bond_representations[3],
                ),
                (self._phys, self._bond_representations[1]),
            )
            p_transpA = p_transpA.transpose(4, 0, 1, 5, 2, 3, 6).reshape(p_data.shape)
            self._isoA2 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._bond_representations[1], self._phys),
                (
                    self._anc,
                    self._bond_representations[0],
                    self._bond_representations[2],
                    self._bond_representations[3],
                ),
            )
            p_transpB = p_transpB.transpose(1, 2, 3, 0, 4, 5, 6).reshape(p_data.shape)
            self._isoB2 = p_transpB.T @ p_data
        return self._isoA2, self._isoB2

    def get_isoAB3(self):
        if self._isoA3 is None:
            if self.verbosity > 1:
                eff3 = self._phys * self._bond_representations[2]
                aux3 = (
                    self._anc
                    * self._bond_representations[0]
                    * self._bond_representations[1]
                    * self._bond_representations[2]
                )
                print(f"compute isoA3 and isoB3: eff_rep3 = {eff3}")
                print(f"aux_rep3 = {aux3}")
            p_data = get_projector_chained(
                self._phys,
                self._anc,
                self._bond_representations[0],
                self._bond_representations[1],
                self._bond_representations[2],
                self._bond_representations[3],
                singlet_only=True,
            )
            p_data = p_data.reshape(-1, p_data.shape[6])
            p_transpA = construct_matrix_projector(
                (
                    self._anc,
                    self._bond_representations[0],
                    self._bond_representations[1],
                    self._bond_representations[3],
                ),
                (self._phys, self._bond_representations[2]),
            )
            p_transpA = p_transpA.transpose(4, 0, 1, 2, 5, 3, 6).reshape(p_data.shape)
            self._isoA3 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._bond_representations[2], self._phys),
                (
                    self._anc,
                    self._bond_representations[0],
                    self._bond_representations[1],
                    self._bond_representations[3],
                ),
            )
            p_transpB = p_transpB.transpose(1, 2, 3, 4, 0, 5, 6).reshape(p_data.shape)
            self._isoB3 = p_transpB.T @ p_data
        return self._isoA3, self._isoB3

    def get_isoAB4(self):
        if self._isoA4 is None:
            if self.verbosity > 1:
                eff4 = self._phys * self._bond_representations[3]
                aux4 = (
                    self._anc
                    * self._bond_representations[0]
                    * self._bond_representations[1]
                    * self._bond_representations[2]
                )
                print(f"compute isoA4 and isoB4: eff_rep4 = {eff4}")
                print(f"aux_rep4 = {aux4}")
            p_data = get_projector_chained(
                self._phys,
                self._anc,
                self._bond_representations[0],
                self._bond_representations[1],
                self._bond_representations[2],
                self._bond_representations[3],
                singlet_only=True,
            )
            p_data = p_data.reshape(-1, p_data.shape[6])
            p_transpA = construct_matrix_projector(
                (
                    self._anc,
                    self._bond_representations[0],
                    self._bond_representations[1],
                    self._bond_representations[2],
                ),
                (self._phys, self._bond_representations[3]),
            )
            p_transpA = p_transpA.transpose(4, 0, 1, 2, 3, 5, 6).reshape(p_data.shape)
            self._isoA4 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._bond_representations[3], self._phys),
                (
                    self._anc,
                    self._bond_representations[0],
                    self._bond_representations[1],
                    self._bond_representations[2],
                ),
            )
            p_transpB = p_transpB.transpose(1, 2, 3, 4, 5, 0, 6).reshape(p_data.shape)
            self._isoB4 = p_transpB.T @ p_data
        return self._isoA4, self._isoB4

    def get_isoAB(self, i):
        if i == 1:
            return self.get_isoAB1()
        if i == 2:
            return self.get_isoAB2()
        if i == 3:
            return self.get_isoAB3()
        if i == 4:
            return self.get_isoAB4()
        raise ValueError

    def update_bond(self, i, gate):
        eff_rep = self._phys * self._bond_representations[i - 1]
        aux_rep = (
            self._anc
            * self._bond_representations[i % 4]
            * self._bond_representations[(i + 1) % 4]
            * self._bond_representations[(i + 2) % 4]
        )
        if self.verbosity > 2:
            print(
                f"update bond {i}: rep {i} = {self._bond_representations[i - 1]},",
                f"aux_rep = {aux_rep}",
            )
        isoA, isoB = self.get_isoAB(i)
        transposedA = isoA @ self._tensors_data[0]
        matA = SU2_Matrix.from_raw_data(transposedA, aux_rep, eff_rep)
        transposedB = isoB @ self._tensors_data[1]
        matB = SU2_Matrix.from_raw_data(transposedB, eff_rep, aux_rep)

        newA, self._weights[i - 1], newB, new_rep = self.update_first_neighbor(
            matA, matB, self._weights[i - 1], self._bond_representations[i - 1], gate
        )
        if new_rep != self._bond_representations[i - 1]:
            self.reset_isometries()
            self._bond_representations[i - 1] = new_rep
            isoA, isoB = self.get_isoAB(i)
        self._tensors_data[0] = isoA.T @ newA.to_raw_data()
        self._tensors_data[1] = isoB.T @ newB.to_raw_data()
