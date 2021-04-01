import numpy as np

from groups.su2_representation import SU2_Representation
from groups.su2_matrix import (
    SU2_Matrix,
    construct_matrix_projector,
    construct_transpose_matrix,
)


class SU2_SimpleUpdate(object):
    """
    SU(2) symmetric simple update algorithm. Only deals with finite temperature. Each
    site has to be an *irreducible* representation of SU(2).

    This class is not made for direct use, unit cell is not defined. Static variables
    _unit_cell, _n_bond, _n_hamilt and_n_tensors must be defined in subclasses.
    """

    _unit_cell = NotImplemented
    _n_bonds = NotImplemented
    _n_hamilts = NotImplemented
    _n_tensors = NotImplemented

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
                f"beta = {beta:.6g}",
            )
            print(f"unit cell:\n{self._unit_cell}")

        self.Dstar = Dstar
        self._beta = beta
        self._phys = SU2_Representation.irrep(self._d)
        self._phys2 = self._phys * self._phys
        self._a = self._d
        self._anc = self._phys
        self._hamilts = [
            SU2_Matrix.from_raw_data(r, self._phys2, self._phys2)
            for r in hamilts_raw_data
        ]
        self.tau = tau
        self.rcutoff = rcutoff

        self._bond_representations = bond_representations
        self._weights = weights
        self._tensors_data = tensors_data
        self.reset_isometries()
        if self.verbosity > 1:
            print(self)

    @classmethod
    def from_infinite_temperature(
        cls, d, Dstar, tau, hamilts, rcutoff=1e-11, verbosity=0
    ):
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
        hamilts : enumerable of (d**2, d**2) ndarray
            Bond Hamltionians. Must be real symmetric or hermitian.
        rcutoff : float, optional.
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        phys = SU2_Representation.irrep(d)
        proj, ind = construct_matrix_projector(
            (phys, phys), (phys, phys), conj_right=True
        )
        h_raw_data = []
        for i, h in enumerate(hamilts):
            if h.shape != (d ** 2, d ** 2):
                raise ValueError(f"invalid shape for Hamiltonian {i}")
            h_raw_data.append(proj.T @ h.ravel()[ind])
        return cls(
            Dstar,
            0.0,
            tau,
            rcutoff,
            h_raw_data,
            [SU2_Representation.irrep(1)] * cls._n_bonds,
            [np.ones(1)] * cls._n_bonds,
            [np.ones(1)] * cls._n_tensors,
            verbosity,
        )

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

    def save_to_file(self, file, additional_data={}):
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

        np.savez_compressed(file, **data, **additional_data)
        if self.verbosity > 0:
            print("Simple update saved in file", file)

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

    def get_dense_weights(self, sort=True):
        """
        Return simple update weights for each bond with degeneracies.
        """
        dense_weights = []
        for (w, rep) in zip(self._weights, self._bond_representations):
            dw = np.empty(rep.dim)
            w_index = 0
            dw_index = 0
            for (deg, irr) in zip(rep.degen, rep.irreps):
                for i in range(deg):
                    dw[dw_index : dw_index + irr] = w[w_index]
                    w_index += 1
                    dw_index += irr
            if sort:
                dw.sort()
                dw = dw[::-1]
            dense_weights.append(dw)
        return dense_weights

    def reset_isometries(self):
        return NotImplemented

    def get_tensors_mz(self):
        """
        Returns
        -------
        tensors : tuple of size _n_tensors
            Optimized dense tensors, with sqrt(weights) on all virtual legs. Virtual
            legs are sorted by weights magnitude, not by SU(2) irreps.
        mz : tuple of size _n_tensors
            mz values for each axis of each tensor.
        """
        return NotImplemented

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
        pL1, pL0 = construct_transpose_matrix(
            (virt_left, self._phys, virt_mid), 1, 2, (0, 1, 2), contract=False
        )
        vL1 = pL1.T @ (pL0 @ effL.to_raw_data())  # faster than contracting pL1 @ pL0
        matL1 = (
            SU2_Matrix.from_raw_data(vL1, virt_left * self._phys, virt_mid) / weights
        )

        pR1, pR0 = construct_transpose_matrix(
            (virt_mid, self._phys, virt_right), 2, 1, (0, 1, 2), contract=False
        )
        vR1 = pR1.T @ (pR0 @ effR.to_raw_data())
        matR1 = SU2_Matrix.from_raw_data(vR1, virt_mid, self._phys * virt_right)

        # construct matrix theta and apply gate
        theta_mat = matL1 @ matR1
        theta = theta_mat.to_raw_data()
        iso_theta = construct_transpose_matrix(
            (virt_left, self._phys, self._phys, virt_right), 2, 2, (0, 3, 1, 2)
        )
        theta2 = iso_theta @ theta
        theta_mat2 = SU2_Matrix.from_raw_data(
            theta2, virt_left * virt_right, self._phys2
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
            pL1, pL0 = construct_transpose_matrix(
                (virt_left, self._phys, new_virt_mid), 1, 2, (0, 1, 2), contract=False
            )
            pR1, pR0 = construct_transpose_matrix(
                (new_virt_mid, self._phys, virt_right), 2, 1, (0, 1, 2), contract=False
            )

        # reshape to initial tree structure
        new_vL = pL0.T @ (pL1 @ U1.to_raw_data())
        new_effL = SU2_Matrix.from_raw_data(
            new_vL, virt_left, self._phys * new_virt_mid
        )
        new_vR = pR0.T @ (pR1 @ V1.to_raw_data())
        new_effR = SU2_Matrix.from_raw_data(
            new_vR, new_virt_mid * self._phys, virt_right
        )

        # reconnect with const parts
        newL = cstL @ new_effL
        newR = new_effR @ cstR

        return newL, newR, new_weights, new_virt_mid

    def update_through_proxy(
        self, matL0, mat_mid0, matR0, weightsL, weightsR, repL, repR, gate
    ):
        r"""
        Apply gate between two tensors through a proxy (either 2nd ord 3rd neighbor)
        A 1D geometry is considered for clarity, the function being direction-agnostic.

        Parameters
        ----------
        matL0 : SU2_Matrix
            "Left" matrix, tree structure is defined below.
        mat_mid0 : SU2_Matrix
            "Middle" matrix, tree structure is defined below.
        matR0 : SU2_Matrix
            "Right" matrix, tree structure is defined below.
        weightsL : numpy array
            Left bond weights before update.
        weightsR : numpy array
            Right bond weights before update.
        repL : SU2_Representation
            Left Bond SU(2) representation before update.
        repR : SU2_Representation
            Right Bond SU(2) representation before update.
        gate : SU2_Matrix
            Gate to apply on the bond. Tree structure must be
            (self._d * self._d, self._d * self._d)


            matL0            mat_mid0         matR0
            /  \             /    \           /  \
           /    \           /\     \         /    \
          /     /\         /  \     \       /\     \
        auxL   p repL    repL repR auxm   repR p  auxR
        """
        # 1) SVD cut between constant tensors and effective tensors to update
        cstL, svL, effL, auxL = matL0.svd()
        effL = svL[:, None] * effL
        eff_m, sv_m, cst_m, aux_m = mat_mid0.svd()
        eff_m = eff_m * sv_m
        effR, svR, cstR, auxR = matR0.svd()
        effR = effR * svR

        # change tensor structure to contract mid
        isoL = construct_transpose_matrix((auxL, self._phys, repL), 1, 2, (0, 1, 2))
        matL1 = (
            SU2_Matrix.from_raw_data(isoL @ effL.to_raw_data(), auxL * self._phys, repL)
            / weightsL
        )

        iso_m = construct_transpose_matrix((repL, repR, aux_m), 2, 1, (0, 1, 2))
        mat_m1 = SU2_Matrix.from_raw_data(
            iso_m @ eff_m.to_raw_data(), repL, repR * aux_m
        )

        isoR = construct_transpose_matrix((repR, self._phys, auxR), 2, 1, (0, 1, 2))
        matR1 = (
            SU2_Matrix.from_raw_data(isoR @ effR.to_raw_data(), repR, self._phys * auxR)
            / weightsR[:, None]
        )

        # contract tensor network
        theta = matL1 @ mat_m1
        iso1 = construct_transpose_matrix(
            (auxL, self._phys, repR, aux_m), 2, 3, (0, 1, 3, 2)
        )
        theta = SU2_Matrix.from_raw_data(
            iso1 @ theta.to_raw_data(), auxL * self._phys * aux_m, repR
        )
        theta = theta @ matR1
        iso2 = construct_transpose_matrix(
            (auxL, self._phys, aux_m, self._phys, auxR), 3, 3, (0, 2, 4, 1, 3)
        )
        theta = SU2_Matrix.from_raw_data(
            iso2 @ theta.to_raw_data(), auxL * aux_m * auxR, self._phys2
        )
        theta = theta @ gate

        # 1st SVD
        theta = SU2_Matrix.from_raw_data(
            iso2.T @ theta.to_raw_data(), auxL * self._phys * aux_m, self._phys * auxR
        )
        U, new_weightsR, V, new_repR = theta.svd(cut=self.Dstar, rcutoff=self.rcutoff)
        new_weightsR /= new_weightsR @ new_repR.get_multiplet_structure()
        effR = V * new_weightsR[:, None]

        # recompute reshape matrices only if needed
        if new_repR != repR:
            isoR = construct_transpose_matrix(
                (new_repR, self._phys, auxR), 2, 1, (0, 1, 2)
            )
            iso1 = construct_transpose_matrix(
                (auxL, self._phys, new_repR, aux_m), 2, 3, (0, 1, 3, 2)
            )

        # cut left from mid
        theta = U * new_weightsR
        theta = SU2_Matrix.from_raw_data(
            iso1.T @ theta.to_raw_data(), auxL * self._phys, new_repR * aux_m
        )
        U, new_weightsL, V, new_repL = theta.svd(cut=self.Dstar, rcutoff=self.rcutoff)
        new_weightsL /= new_weightsL @ new_repL.get_multiplet_structure()
        eff_m = V * new_weightsL[:, None]
        effL = U * new_weightsL
        if new_repL != repL:
            isoL = construct_transpose_matrix(
                (auxL, self._phys, new_repL), 1, 2, (0, 1, 2)
            )
            iso_m = construct_transpose_matrix(
                (new_repL, new_repR, aux_m), 2, 1, (0, 1, 2)
            )
        elif new_repR != repR:
            iso_m = construct_transpose_matrix(
                (new_repL, new_repR, aux_m), 2, 1, (0, 1, 2)
            )

        # reshape to initial tree structure
        effL = SU2_Matrix.from_raw_data(
            isoL.T @ effL.to_raw_data(), auxL, new_repL * self._phys
        )
        eff_m = SU2_Matrix.from_raw_data(
            iso_m.T @ eff_m.to_raw_data(), new_repL * new_repR, aux_m
        )
        effR = SU2_Matrix.from_raw_data(
            isoR.T @ effR.to_raw_data(), new_repR * self._phys, auxR
        )

        # reconnect with const parts
        newL = cstL @ effL
        new_mid = eff_m @ cst_m
        newR = effR @ cstR
        return newL, new_mid, newR, new_weightsL, new_weightsR, new_repL, new_repR


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

    Notice that leg index starts at 1, a -1 shift is required to get array index.
    """

    _unit_cell = "AB"
    _n_bonds = 4
    _n_hamilts = 1
    _n_tensors = 2

    # There are only 2 tensors, which have exactly the same bonds. Hence we can use the
    # same isometry to move their axes and give them the same tree structure, with the
    # ancilla + 3 non-updates legs as rows and physical leg + updated leg as columns,
    # then just transpose matB. Due to cyclical 2nd order Trotter-Suzuki, only 3
    # isometries are required.

    # permutations used in get_isometry
    #                       U -> R               R -> D            D -> L
    _isometry_swaps = ((2, 1, 0, 3, 4, 5), (3, 1, 2, 0, 4, 5), (4, 1, 2, 3, 0, 5))

    def __repr__(self):
        return f"SU2_SimpleUpdate1x2 for irrep {self._d}"

    def __str__(self):
        return (
            f"SU2_SimpleUpdate1x2 for irrep {self._d} at beta = {self._beta:.6g}\n"
            f"D* = {self.Dstar}, tau = {self._tau}, rcutoff = {self.rcutoff}\n"
            f"bond 1 representation: {self._bond_representations[0]}\n"
            f"bond 2 representation: {self._bond_representations[1]}\n"
            f"bond 3 representation: {self._bond_representations[2]}\n"
            f"bond 4 representation: {self._bond_representations[3]}"
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
        if beta_evolve < -1e-12:
            raise ValueError("Cannot evolve for negative imaginary time")
        if beta_evolve < 0.9 * self._dbeta:  # care for float rounding
            return  # else evolve for 1 step out of niter loop
        niter = round(beta_evolve / self._dbeta)  # 2nd order: evolve 2*tau by step

        self._update_bond(1, self._gates[0], backwards=True)
        for i in range(niter - 1):  # there is 1 step out of the loop
            self._update_bond(2, self._gates[0])
            self._update_bond(3, self._gates[0])
            self._update_bond(4, self._squared_gates[0])
            self._update_bond(3, self._gates[0], backwards=True)
            self._update_bond(2, self._gates[0], backwards=True)
            self._update_bond(1, self._squared_gates[0], backwards=True)
            self._beta += self._dbeta
        self._update_bond(2, self._gates[0])
        self._update_bond(3, self._gates[0])
        self._update_bond(4, self._squared_gates[0])
        self._update_bond(3, self._gates[0], backwards=True)
        self._update_bond(2, self._gates[0], backwards=True)
        self._update_bond(1, self._gates[0], backwards=True)
        self._reset_tensor_state()  # always end up ready for update bond 1
        self._beta += self._dbeta

    def _reset_tensor_state(self):
        """
        Apply isoA/B_{1->2} to allow for two bond 1 updates in a row.
        """
        iso = self.get_isometry(2)
        self._tensors_data[0] = iso @ self._tensors_data[0]
        self._tensors_data[1] = iso @ self._tensors_data[1]

    def reset_isometries(self):
        if self.verbosity > 1:
            print(f"reset isometries at beta = {self._beta:.6g}")
        # 3 isometries: up->right, right->down, down->left. Same for A and B.
        self._isometries = [None] * 3

    def get_tensors_mz(self):
        """
        Returns:
        -------
        (A, B) : tuple of 2 ndarrays
            Optimized dense tensors, with sqrt(weights) on all virtual legs. Virtual
            legs are sorted by weights magnitude, not by SU(2) irreps.
        (colorsA, colorsB) : tuple of tuple
            Sz eigenvalues for each axis of each tensor.
        """
        w1, w2, w3, w4 = [1.0 / np.sqrt(w) for w in self.get_dense_weights(sort=False)]
        so1 = w1.argsort()
        so2 = w2.argsort()
        so3 = w3.argsort()
        so4 = w4.argsort()
        sz_val0 = self._phys.get_Sz()
        sz_val1 = self._bond_representations[0].get_Sz()[so1]
        sz_val2 = self._bond_representations[1].get_Sz()[so2]
        sz_val3 = self._bond_representations[2].get_Sz()[so3]
        sz_val4 = self._bond_representations[3].get_Sz()[so4]

        D1 = w1.size
        D2 = w2.size
        D3 = w3.size
        D4 = w4.size
        size = self._d * self._a * D1 * D2 * D3 * D4

        # su must be in state 1, after an update on bond 1. This is always true after
        # an evolve call.
        aux_rep = (
            self._bond_representations[1],
            self._bond_representations[2],
            self._bond_representations[3],
            self._anc,
        )
        eff_rep = (self._bond_representations[0], self._phys)
        proj, indices = construct_matrix_projector(eff_rep, aux_rep)
        gammaA = np.zeros(size)
        gammaA[indices] = proj @ self._tensors_data[0]
        gammaA = gammaA.reshape(D1, self._d, D2, D3, D4, self._a)
        gammaA = np.einsum("uprdla,u,r,d,l->paurdl", gammaA, w1, w2, w3, w4)
        gammaA = gammaA[
            :, :, so1[:, None, None, None], so2[:, None, None], so3[:, None], so4
        ]
        gammaA /= np.amax(gammaA)

        projB, indicesB = construct_matrix_projector(eff_rep[::-1], aux_rep)
        gammaB = np.zeros(size)
        gammaB[indices] = proj @ self._tensors_data[1]
        gammaB = gammaB.reshape(D1, self._d, D2, D3, D4, self._a)
        gammaB = np.einsum("dplura,u,r,d,l->paurdl", gammaB, w3, w4, w1, w2)
        gammaB = gammaB[
            :, :, so3[:, None, None, None], so4[:, None, None], so1[:, None], so2
        ]
        gammaB /= np.amax(gammaB)
        return (
            (gammaA, gammaB),
            (
                (sz_val0, sz_val0, sz_val1, sz_val2, sz_val3, sz_val4),
                (-sz_val0, -sz_val0, -sz_val3, -sz_val4, -sz_val1, -sz_val2),
            ),
        )

    def get_isometry(self, i, backwards=False):
        ind = i + backwards - 2
        if self._isometries[ind] is None:
            self.construct_isometry(ind)
        if backwards:
            return self._isometries[ind].T
        return self._isometries[ind]

    def construct_isometry(self, ind):
        if self.verbosity > 1:
            print(f"compute isometry for direction {ind}")
            print(*self._bond_representations, sep="\n")
        # function works for any direction, for simplicity assume bond 1 in variables
        leg_indices = sorted([(ind + 1) % 4, (ind + 2) % 4, (ind + 3) % 4])
        rep1 = self._bond_representations[ind]
        rep2 = self._bond_representations[leg_indices[0]]
        rep3 = self._bond_representations[leg_indices[1]]
        rep4 = self._bond_representations[leg_indices[2]]

        self._isometries[ind] = construct_transpose_matrix(
            (rep1, self._phys, rep2, rep3, rep4, self._anc),
            2,
            2,
            self._isometry_swaps[ind],
        )

    def _update_bond(self, i, gate, backwards=False):
        """
        Update bond i between tensors A and B.
        Tranpose tensors A and B with isoAB matrices. The isometry used depends on
        current tensor tree structure, which was set by last update. This update is
        either i - 1 or i + 1 if backwards.

        This method is not exposed to prevent tensors entering an ill-defined state
        where last update is neither i - 1 or i + 1. It must be called through evolve,
        which ensures it.
        Another solution would be to remember current state of tensors, not needed.
        """
        # bond indices start at 1: -1 shit to get corresponding element in array
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
        iso = self.get_isometry(i, backwards)
        transposedA = iso @ self._tensors_data[0]
        matA = SU2_Matrix.from_raw_data(transposedA, eff_rep, aux_rep).T
        transposedB = iso @ self._tensors_data[1]
        matB = SU2_Matrix.from_raw_data(transposedB, eff_rep, aux_rep)

        (newA, newB, self._weights[i - 1], new_rep) = self.update_first_neighbor(
            matA, matB, self._weights[i - 1], self._bond_representations[i - 1], gate
        )
        if new_rep != self._bond_representations[i - 1]:
            self.reset_isometries()
            self._bond_representations[i - 1] = new_rep

        self._tensors_data[0] = newA.T.to_raw_data()
        self._tensors_data[1] = newB.to_raw_data()


class SU2_SimpleUpdate2x2(SU2_SimpleUpdate):
    """
    SU(2) symmetric simple update algorithm on plaquette AB//CD, with
    - 8 bonds
    - 2 Hamiltonian
    - 4 tensors

    Conventions for leg ordering
    ----------------------------
          1     5
          |     |
       4--A--2--B--4
          |     |
          3     6
          |     |
       8--C--7--D--8
          |     |
          1     5

    Notice that leg index starts at 1, a -1 shift is required to get array index.
    """

    _unit_cell = "AB\nCD"
    _n_bonds = 8
    _n_hamilts = 2
    _n_tensors = 4

    _tensor_legs = ((0, 1, 2, 3), (4, 3, 5, 1), (2, 6, 0, 7), (5, 7, 4, 6))
    _isometry_swaps = (
        (0, 1, 2, 3, 4, 5),  # default to up update
        (2, 1, 0, 3, 4, 5),  # default to right update
        (3, 1, 0, 2, 4, 5),  # default to down update
        (4, 1, 0, 2, 3, 5),  # default to left update
        (0, 2, 1, 3, 4, 5),  # default to UR proxy
        (2, 3, 1, 0, 4, 5),  # default to RD proxy
        (3, 4, 1, 0, 2, 5),  # default to DL proxy
        (4, 0, 1, 2, 3, 5),  # default to LU proxy
    )

    def __repr__(self):
        return f"SU2_SimpleUpdate2x2 for irrep {self._d}"

    def __str__(self):
        return (
            f"SU2_SimpleUpdate2x2 for irrep {self._d} at beta = {self._beta:.6g}\n"
            f"D* = {self.Dstar}, tau = {self._tau}, rcutoff = {self.rcutoff}\n"
            f"bond 1 representation: {self._bond_representations[0]}\n"
            f"bond 2 representation: {self._bond_representations[1]}\n"
            f"bond 3 representation: {self._bond_representations[2]}\n"
            f"bond 4 representation: {self._bond_representations[3]}\n"
            f"bond 5 representation: {self._bond_representations[4]}\n"
            f"bond 6 representation: {self._bond_representations[5]}\n"
            f"bond 7 representation: {self._bond_representations[6]}\n"
            f"bond 8 representation: {self._bond_representations[7]}"
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
        if beta_evolve < -1e-12:
            raise ValueError("Cannot evolve for negative imaginary time")
        if beta_evolve < 0.9 * self._dbeta:  # care for float rounding
            return  # else evolve for 1 step out of niter loop
        niter = round(beta_evolve / self._dbeta)  # 2nd order: evolve 2*tau by step

        self.update_bond1(self._gates[0])
        for i in range(niter - 1):  # there is 1 step out of the loop
            self._2nd_order_step_no1()
            self.update_bond1(self._squared_gates[0])
            self._beta += self._dbeta
            if self.verbosity > 2:
                print(f"set beta to {self._beta:.6g}")
        self._2nd_order_step_no1()
        self.update_bond1(self._gates[0])
        self._beta += self._dbeta

    def _2nd_order_step_no1(self):
        self.update_bond2(self._gates[0])
        self.update_bond3(self._gates[0])
        self.update_bond4(self._gates[0])
        self.update_bond5(self._gates[0])
        self.update_bond6(self._gates[0])
        self.update_bond7(self._gates[0])
        self.update_bond8(self._gates[0])
        self.update_bonds18(self._sqrt_gates[1])
        self.update_bonds54(self._sqrt_gates[1])
        self.update_bonds71(self._sqrt_gates[1])
        self.update_bonds25(self._sqrt_gates[1])
        self.update_bonds62(self._sqrt_gates[1])
        self.update_bonds37(self._sqrt_gates[1])
        self.update_bonds83(self._sqrt_gates[1])
        self.update_bonds46(self._sqrt_gates[1])
        self.update_bonds57(self._sqrt_gates[1])
        self.update_bonds12(self._sqrt_gates[1])
        self.update_bonds41(self._sqrt_gates[1])
        self.update_bonds85(self._sqrt_gates[1])
        self.update_bonds68(self._sqrt_gates[1])
        self.update_bonds34(self._sqrt_gates[1])
        self.update_bonds23(self._sqrt_gates[1])
        self.update_bonds76(self._gates[1])
        self.update_bonds23(self._sqrt_gates[1])
        self.update_bonds34(self._sqrt_gates[1])
        self.update_bonds68(self._sqrt_gates[1])
        self.update_bonds85(self._sqrt_gates[1])
        self.update_bonds41(self._sqrt_gates[1])
        self.update_bonds12(self._sqrt_gates[1])
        self.update_bonds57(self._sqrt_gates[1])
        self.update_bonds46(self._sqrt_gates[1])
        self.update_bonds83(self._sqrt_gates[1])
        self.update_bonds37(self._sqrt_gates[1])
        self.update_bonds62(self._sqrt_gates[1])
        self.update_bonds25(self._sqrt_gates[1])
        self.update_bonds71(self._sqrt_gates[1])
        self.update_bonds54(self._sqrt_gates[1])
        self.update_bonds18(self._sqrt_gates[1])
        self.update_bond8(self._gates[0])
        self.update_bond7(self._gates[0])
        self.update_bond6(self._gates[0])
        self.update_bond5(self._gates[0])
        self.update_bond4(self._gates[0])
        self.update_bond3(self._gates[0])
        self.update_bond2(self._gates[0])

    def reset_isometries(self):
        self._isometries = [[None] * 8 for i in range(self._n_tensors)]

    def reset_isometries_tensor(self, ti):
        if self.verbosity > 1:
            print(f"reset isometries for tensor {ti} at beta = {self._beta:.6g}")
        # each tensor has 4 update states (URDL) and 4 proxy states (UR, RD, DL, LU)
        # isometries are reset only for tensor ti
        self._isometries[ti] = [None] * 8

    def get_tensors_mz(self):
        """
        Returns
        -------
        (A, B, C, D) : tuple of 4 ndarrays
            Optimized dense tensors, with sqrt(weights) on all virtual legs. Virtual
            legs are sorted by weights magnitude, not by SU(2) irreps.
        (colorsA, colorsB, colorsC, colorsD) : tuple of tuple
            Sz eigenvalues for each axis of each tensor.
        """
        sqweights = [1.0 / np.sqrt(w) for w in self.get_dense_weights(sort=False)]
        Ds = np.array([sqw.size for sqw in sqweights])
        so_all = [sqw.argsort() for sqw in sqweights]
        sz_vals = [self._phys.get_Sz()] + [
            rep.get_Sz()[so] for (rep, so) in zip(self._bond_representations, so_all)
        ]

        size = self._d * self._a * Ds.prod()
        # TODO
        sz_vals, size

    def get_isometry(self, tensor, direction):
        if self._isometries[tensor][direction] is None:
            rep1 = self._bond_representations[self._tensor_legs[tensor][0]]
            rep2 = self._bond_representations[self._tensor_legs[tensor][1]]
            rep3 = self._bond_representations[self._tensor_legs[tensor][2]]
            rep4 = self._bond_representations[self._tensor_legs[tensor][3]]
            if self.verbosity > 1:
                print(f"Compute isometry for tensor {tensor} and direction {direction}")
                print(f"rep{self._tensor_legs[tensor][0] + 1} = {rep1}")
                print(f"rep{self._tensor_legs[tensor][1] + 1} = {rep2}")
                print(f"rep{self._tensor_legs[tensor][2] + 1} = {rep3}")
                print(f"rep{self._tensor_legs[tensor][3] + 1} = {rep4}")
            self._isometries[tensor][direction] = construct_transpose_matrix(
                (rep1, self._phys, rep2, rep3, rep4, self._anc),
                3,
                2,
                self._isometry_swaps[direction],
            )
        return self._isometries[tensor][direction]

    def _update_bond_i(self, gate, iA, iC, dirA):
        """
        Generic first neighbor update function for plaquette AB//CD. Variable names
        follow update_bond1 conventions.

        Parameters
        ----------
        iA : int
            Index of left tensor: 0 for A, 1 for B, 2 for C, 3 for D.
        iC : int
            Index of right tensor.
        dirA : int
            Updated direction for left tensor: 0 for up, 1 for right, 2 for down, 3 for
            left.
        """
        dirC = (dirA + 2) % 4
        i1 = self._tensor_legs[iA][dirA]  # index of updated leg
        eff_rep = self._phys * self._bond_representations[i1]
        shared = self._anc * self._bond_representations[self._tensor_legs[iA][dirC]]
        aux_repA = (
            shared
            * self._bond_representations[self._tensor_legs[iA][(dirA + 1) % 4]]
            * self._bond_representations[self._tensor_legs[iA][(dirA + 3) % 4]]
        )
        aux_repC = (
            shared
            * self._bond_representations[self._tensor_legs[iC][(dirA + 1) % 4]]
            * self._bond_representations[self._tensor_legs[iC][(dirA + 3) % 4]]
        )
        if self.verbosity > 2:
            print(f"update bond {i1+1}: rep{i1+1} = {self._bond_representations[i1]}")
            print(f"1st aux_rep = {aux_repA}")
            print(f"2nd aux_rep = {aux_repC}")
        isoA = self.get_isometry(iA, dirA)
        isoC = self.get_isometry(iC, dirC)
        matA = SU2_Matrix.from_raw_data(
            isoA @ self._tensors_data[iA], eff_rep, aux_repA
        ).T
        matC = SU2_Matrix.from_raw_data(
            isoC @ self._tensors_data[iC], eff_rep, aux_repC
        )

        newA, newC, self._weights[i1], new_rep = self.update_first_neighbor(
            matA, matC, self._weights[i1], self._bond_representations[i1], gate
        )

        if new_rep != self._bond_representations[i1]:
            self._bond_representations[i1] = new_rep
            self.reset_isometries_tensor(iA)
            self.reset_isometries_tensor(iC)
            isoA = self.get_isometry(iA, dirA)
            isoC = self.get_isometry(iC, dirC)

        self._tensors_data[iA] = isoA.T @ newA.T.to_raw_data()
        self._tensors_data[iC] = isoC.T @ newC.to_raw_data()

    # leg indices have a -1 shift to start at 0.
    def update_bond1(self, gate):
        self._update_bond_i(gate, 0, 2, 0)

    def update_bond2(self, gate):
        self._update_bond_i(gate, 0, 1, 1)

    def update_bond3(self, gate):
        self._update_bond_i(gate, 0, 2, 2)

    def update_bond4(self, gate):
        self._update_bond_i(gate, 0, 1, 3)

    def update_bond5(self, gate):
        self._update_bond_i(gate, 1, 3, 0)

    def update_bond6(self, gate):
        self._update_bond_i(gate, 1, 3, 2)

    def update_bond7(self, gate):
        self._update_bond_i(gate, 2, 3, 1)

    def update_bond8(self, gate):
        self._update_bond_i(gate, 2, 3, 3)

    def _update_second_neighbor(self, gate, iA, iB, iD, dirsB):
        """
        Generic second neighbor update function for plaquette AB//CD. Variable names
        follow update 2-5 conventions.

        Parameters
        ----------
        iA : int
            Index of "left" tensor: 0 for A, 1 for B, 2 for C, 3 for D.
        iB : int
            Index of proxy tensor.
        iD : int
            Index of "right" tensor.
        dirsB : int
            Updated directions for proxy tensor:
            - 4 for up-right
            - 5 for right-down
            - 6 for down-left
            - 7 for left-up.
        """
        dirA = (dirsB + 2) % 4  # # direction for tensor "A"
        dirD = (dirsB + 3) % 4  # # direction for tensor "D"
        i2 = self._tensor_legs[iA][dirA]  # index of 1st updated leg
        i5 = self._tensor_legs[iD][dirD]  # index of 2nd updated leg
        eff_repA = self._phys * self._bond_representations[i2]
        eff_repD = self._phys * self._bond_representations[i5]
        eff_repB = self._bond_representations[i2] * self._bond_representations[i5]
        aux_repB = (
            self._phys2  # phys * anc
            * self._bond_representations[self._tensor_legs[iB][dirA]]
            * self._bond_representations[self._tensor_legs[iB][dirD]]
        )
        aux_repA = (
            self._anc
            * self._bond_representations[self._tensor_legs[iA][(dirA + 1) % 4]]
            * self._bond_representations[self._tensor_legs[iA][(dirA + 2) % 4]]
            * self._bond_representations[self._tensor_legs[iA][(dirA + 3) % 4]]
        )
        aux_repD = (
            self._anc
            * self._bond_representations[self._tensor_legs[iD][(dirD + 1) % 4]]
            * self._bond_representations[self._tensor_legs[iD][(dirD + 2) % 4]]
            * self._bond_representations[self._tensor_legs[iD][(dirD + 3) % 4]]
        )
        if self.verbosity > 2:
            print(f"update bonds {i2+1} and {i5+1}:")
            print(f"rep{i2+1} = {self._bond_representations[i2]}")
            print(f"rep{i5+1} = {self._bond_representations[i5]}")
            print(f"left aux_rep = {aux_repA}")
            print(f"mid aux_rep = {aux_repB}")
            print(f"right aux_rep = {aux_repD}")
        isoA = self.get_isometry(iA, dirA)
        isoB = self.get_isometry(iB, dirsB)
        isoD = self.get_isometry(iD, dirD)
        matA = SU2_Matrix.from_raw_data(
            isoA @ self._tensors_data[iA], eff_repA, aux_repA
        ).T
        matB = SU2_Matrix.from_raw_data(
            isoB @ self._tensors_data[iB], eff_repB, aux_repB
        )
        matD = SU2_Matrix.from_raw_data(
            isoD @ self._tensors_data[iD], eff_repD, aux_repD
        )

        (
            newA,
            newB,
            newD,
            self._weights[i2],
            self._weights[i5],
            new_rep2,
            new_rep5,
        ) = self.update_through_proxy(
            matA,
            matB,
            matD,
            self._weights[i2],
            self._weights[i5],
            self._bond_representations[i2],
            self._bond_representations[i5],
            gate,
        )

        if new_rep2 != self._bond_representations[i2]:
            self._bond_representations[i2] = new_rep2
            self.reset_isometries_tensor(iA)
            self.reset_isometries_tensor(iB)
            isoA = self.get_isometry(iA, dirA)
        if new_rep5 != self._bond_representations[i5]:
            self._bond_representations[i5] = new_rep5
            self.reset_isometries_tensor(iB)
            self.reset_isometries_tensor(iD)
            isoD = self.get_isometry(iD, dirD)

        isoB = self.get_isometry(iB, dirsB)
        self._tensors_data[iA] = isoA.T @ newA.T.to_raw_data()
        self._tensors_data[iB] = isoB.T @ newB.to_raw_data()
        self._tensors_data[iD] = isoD.T @ newD.to_raw_data()

    def update_bonds18(self, gate):
        self._update_second_neighbor(gate, 0, 2, 3, 6)

    def update_bonds54(self, gate):
        self._update_second_neighbor(gate, 3, 1, 0, 4)

    def update_bonds71(self, gate):
        self._update_second_neighbor(gate, 3, 2, 0, 5)

    def update_bonds25(self, gate):
        self._update_second_neighbor(gate, 0, 1, 3, 7)

    def update_bonds62(self, gate):
        self._update_second_neighbor(gate, 3, 1, 0, 6)

    def update_bonds37(self, gate):
        self._update_second_neighbor(gate, 0, 2, 3, 4)

    def update_bonds83(self, gate):
        self._update_second_neighbor(gate, 3, 2, 0, 7)

    def update_bonds46(self, gate):
        self._update_second_neighbor(gate, 0, 1, 3, 5)

    def update_bonds57(self, gate):
        self._update_second_neighbor(gate, 1, 3, 2, 6)

    def update_bonds12(self, gate):
        self._update_second_neighbor(gate, 2, 0, 1, 4)

    def update_bonds41(self, gate):
        self._update_second_neighbor(gate, 1, 0, 2, 7)

    def update_bonds85(self, gate):
        self._update_second_neighbor(gate, 2, 3, 1, 5)

    def update_bonds68(self, gate):
        self._update_second_neighbor(gate, 1, 3, 2, 4)

    def update_bonds34(self, gate):
        self._update_second_neighbor(gate, 2, 0, 1, 6)

    def update_bonds23(self, gate):
        self._update_second_neighbor(gate, 1, 0, 2, 5)

    def update_bonds76(self, gate):
        self._update_second_neighbor(gate, 2, 3, 1, 7)
