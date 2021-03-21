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

    def get_tensors_mz(self):
        """
        Returns:
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
            self.reset_isometries()
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

    Notice that leg index starts at 1, a -1 shift is required to get array index.
    """

    _unit_cell = "AB"
    _n_bonds = 4
    _n_hamilts = 1
    _n_tensors = 2

    # transpositions used in get_isoAB
    _isoA_swaps = ((5, 1, 2, 3, 4, 0), (0, 5, 2, 3, 4, 1), (0, 1, 5, 3, 4, 2))
    _isoB_swaps = ((2, 1, 0, 3, 4, 5), (3, 1, 2, 0, 4, 5), (4, 1, 2, 3, 0, 5))

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
        proj, ind = construct_matrix_projector(
            (phys, phys), (phys, phys), conj_right=True
        )
        h_raw_data = proj.T @ h.ravel()[ind]
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
        isoA, isoB = self.get_isoAB(2)
        self._tensors_data[0] = isoA @ self._tensors_data[0]
        self._tensors_data[1] = isoB @ self._tensors_data[1]

    def reset_isometries(self):
        if self.verbosity > 1:
            print(f"reset isometries at beta = {self._beta:.6g}")
        # 3 isometries: 1<->2, 2<->3 and 3<->4
        self._isoA = [None] * 3
        self._isoB = [None] * 3

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
        eff_rep = (self._phys, self._bond_representations[0])
        projA, indicesA = construct_matrix_projector(aux_rep, eff_rep)
        gammaA = np.zeros(size)
        gammaA[indicesA] = projA @ self._tensors_data[0]
        del projA, indicesA
        gammaA = gammaA.reshape(D2, D3, D4, self._a, self._d, D1)
        gammaA = np.einsum("rdlapu,u,r,d,l->paurdl", gammaA, w1, w2, w3, w4)
        gammaA = gammaA[
            :, :, so1[:, None, None, None], so2[:, None, None], so3[:, None], so4
        ]
        gammaA /= np.amax(gammaA)

        projB, indicesB = construct_matrix_projector(eff_rep[::-1], aux_rep)
        gammaB = np.zeros(size)
        gammaB[indicesB] = projB @ self._tensors_data[1]
        del projB, indicesB
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

    def get_isoAB(self, i, backwards=False):
        ind = i + backwards - 2
        assert -1 < ind < 3
        assert i != 1 or backwards
        assert i != 4 or not backwards
        if self._isoA[ind] is None:
            if self.verbosity > 1:
                print(f"compute isoA and isoB for move {i-1+2*backwards} -> {i}")
                print(*self._bond_representations, sep="\n")

            # function works for any bond, for simplicity assume bond 1 in variables
            leg_indices = sorted([(ind + 1) % 4, (ind + 2) % 4, (ind + 3) % 4])
            rep1 = self._bond_representations[ind]
            rep2 = self._bond_representations[leg_indices[0]]
            rep3 = self._bond_representations[leg_indices[1]]
            rep4 = self._bond_representations[leg_indices[2]]

            self._isoA[ind] = construct_transpose_matrix(
                (rep2, rep3, rep4, self._anc, self._phys, rep1),
                4,
                4,
                self._isoA_swaps[ind],
            )
            self._isoB[ind] = construct_transpose_matrix(
                (rep1, self._phys, rep2, rep3, rep4, self._anc),
                2,
                2,
                self._isoB_swaps[ind],
            )

        if backwards:
            return self._isoA[ind].T, self._isoB[ind].T
        return self._isoA[ind], self._isoB[ind]

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
        isoA, isoB = self.get_isoAB(i, backwards)
        transposedA = isoA @ self._tensors_data[0]
        matA = SU2_Matrix.from_raw_data(transposedA, aux_rep, eff_rep)
        transposedB = isoB @ self._tensors_data[1]
        matB = SU2_Matrix.from_raw_data(transposedB, eff_rep, aux_rep)

        (
            newA,
            self._weights[i - 1],
            newB,
            self._bond_representations[i - 1],
        ) = self.update_first_neighbor(
            matA, matB, self._weights[i - 1], self._bond_representations[i - 1], gate
        )

        self._tensors_data[0] = newA.to_raw_data()
        self._tensors_data[1] = newB.to_raw_data()
