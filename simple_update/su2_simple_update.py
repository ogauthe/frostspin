import numpy as np

from groups.su2_representation import (
    SU2_Representation,
    SU2_Matrix,
    get_projector_chained,
    construct_matrix_projector,
)


class SU2_SimpleUpdate1x2(object):
    """
    SU(2) symmetric simple update algorithm on plaquette AB. Only deals with finite
    temperature. Each site has to be an *irreducible* representation of SU(2).

    Conventions for leg ordering
    ----------------------------
          1     3
          |     |
       4--A--2--B--4
          |     |
          3     1
    """

    def __init__(
        self,
        Dstar,
        beta,
        h_raw_data,
        tau,
        rep1,
        rep2,
        rep3,
        rep4,
        weights1,
        weights2,
        weights3,
        weights4,
        dataA,
        dataB,
        rcutoff,
        verbosity,
    ):
        """
        Initialize SU2_SimpleUpdate1x2 from values. Not meant for direct use, consider
        calling from_infinite_temperature or from_file class methods.
        """
        self.verbosity = verbosity
        self._d = h_raw_data.size
        if self.verbosity > 0:
            print(
                f"Construct SU2_SimpleUpdata1x2 with d = {self._d}, D* = {Dstar},",
                f"beta = {beta}",
            )

        self.Dstar = Dstar
        self._beta = beta
        self._phys = SU2_Representation.irrep(self._d)
        self._a = self._d
        self._anc = self._phys
        d2 = self._phys * self._phys
        self._h = SU2_Matrix.from_raw_data(h_raw_data, d2, d2)
        self.tau = tau

        self._rep1 = rep1
        self._rep2 = rep2
        self._rep3 = rep3
        self._rep4 = rep4
        self._weights1 = weights1
        self._weights2 = weights2
        self._weights3 = weights3
        self._weights4 = weights4
        self._dataA = dataA
        self._dataB = dataB
        self.rcutoff = rcutoff
        if self.verbosity > 1:
            print(f"rep1 = {rep1}")
            print(f"rep2 = {rep2}")
            print(f"rep3 = {rep3}")
            print(f"rep4 = {rep4}")
        self.reset_isometries()

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
            h_raw_data,
            tau,
            SU2_Representation.irrep(1),
            SU2_Representation.irrep(1),
            SU2_Representation.irrep(1),
            SU2_Representation.irrep(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            rcutoff,
            verbosity,
        )

    @classmethod
    def from_file(cls, file, verbosity=0):
        """
        Load simple update from given file.

        Parameters
        ----------
        file : str, optional
            Save file containing data to restart computation from. Must follow
            save_to_file syntax.
        verbosity : int
          Level of log verbosity. Default is no log.
        """
        if verbosity > 0:
            print("Restart SU2_SimpleUpdate1x2 back from file", file)
        with np.load(file) as data:
            Dstar = data["_SU2_SU1x2_Dstar"][()]
            beta = data["_SU2_SU1x2_beta"][()]
            tau = data["_SU2_SU1x2_tau"][()]
            h_raw_data = data["_SU2_SU1x2_h_raw_data"]
            rep1_degen = data["_SU2_SU1x2_rep1_degen"]
            rep1_irreps = data["_SU2_SU1x2_rep1_irreps"]
            rep2_degen = data["_SU2_SU1x2_rep2_degen"]
            rep2_irreps = data["_SU2_SU1x2_rep2_irreps"]
            rep3_degen = data["_SU2_SU1x2_rep3_degen"]
            rep3_irreps = data["_SU2_SU1x2_rep3_irreps"]
            rep4_degen = data["_SU2_SU1x2_rep4_degen"]
            rep4_irreps = data["_SU2_SU1x2_rep4_irreps"]
            weights1 = data["_SU2_SU1x2_weights1"]
            weights2 = data["_SU2_SU1x2_weights2"]
            weights3 = data["_SU2_SU1x2_weights3"]
            weights4 = data["_SU2_SU1x2_weights4"]
            dataA = data["_SU2_SU1x2_dataA"]
            dataB = data["_SU2_SU1x2_dataB"]
            rcutoff = data["_SU2_SU1x2_rcutoff"][()]

        rep1 = SU2_Representation(rep1_degen, rep1_irreps)
        rep2 = SU2_Representation(rep2_degen, rep2_irreps)
        rep3 = SU2_Representation(rep3_degen, rep3_irreps)
        rep4 = SU2_Representation(rep4_degen, rep4_irreps)

        return cls(
            Dstar,
            beta,
            h_raw_data,
            tau,
            rep1,
            rep2,
            rep3,
            rep4,
            weights1,
            weights2,
            weights3,
            weights4,
            dataA,
            dataB,
            rcutoff,
            verbosity,
        )

    def save_to_file(self, file):
        data = {}
        data["_SU2_SU1x2_Dstar"] = self.Dstar
        data["_SU2_SU1x2_beta"] = self._beta
        data["_SU2_SU1x2_tau"] = self._tau
        data["_SU2_SU1x2_h_raw_data"] = self._h.to_raw_data()
        data["_SU2_SU1x2_rep1_degen"] = self._rep1.degen
        data["_SU2_SU1x2_rep1_irreps"] = self._rep1.irreps
        data["_SU2_SU1x2_rep2_degen"] = self._rep2.degen
        data["_SU2_SU1x2_rep2_irreps"] = self._rep2.irreps
        data["_SU2_SU1x2_rep3_degen"] = self._rep3.degen
        data["_SU2_SU1x2_rep3_irreps"] = self._rep3.irreps
        data["_SU2_SU1x2_rep4_degen"] = self._rep4.degen
        data["_SU2_SU1x2_rep4_irreps"] = self._rep4.irreps
        data["_SU2_SU1x2_weights1"] = self._weights1
        data["_SU2_SU1x2_weights2"] = self._weights2
        data["_SU2_SU1x2_weights3"] = self._weights3
        data["_SU2_SU1x2_weights4"] = self._weights4
        data["_SU2_SU1x2_dataA"] = self._dataA
        data["_SU2_SU1x2_dataB"] = self._dataA
        data["_SU2_SU1x2_rcutoff"] = self.rcutoff
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
        self._gate = (-tau * self._h).expm()
        self._gate_squared = (-2 * tau * self._h).expm()

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
        Compute the entanglement entropy on every bonds as S = -sum_i p_i log_pi
        """
        s = np.empty(4)
        s[0] = -(
            self._weights1
            * np.log(self._weights1)
            @ self._rep1.get_multiplet_structure()
        )
        s[1] = -(
            self._weights2
            * np.log(self._weights2)
            @ self._rep2.get_multiplet_structure()
        )
        s[2] = -(
            self._weights3
            * np.log(self._weights3)
            @ self._rep3.get_multiplet_structure()
        )
        s[3] = -(
            self._weights3
            * np.log(self._weights4)
            @ self._rep4.get_multiplet_structure()
        )
        return s

    def get_AB(self):
        """
        Return optimized tensors A and B.
        Tensors are obtained by adding relevant sqrt(lambda) to every leg of gammaX
        For each virtual axis, sort by decreasing weights (instead of SU(2) order)
        """
        # actually weights are on by default, so *remove* sqrt(lambda)
        sl1 = 1 / np.sqrt(self._weights1)
        sl2 = 1 / np.sqrt(self._weights2)
        sl3 = 1 / np.sqrt(self._weights3)
        sl4 = 1 / np.sqrt(self._weights4)
        # TODO add multiplicities
        A = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaA, sl1, sl2, sl3, sl4)
        B = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaB, sl3, sl4, sl1, sl2)
        return A, B

    def evolve(self, beta=None):
        """
        Evolve in imaginary time using second order Trotter-Suzuki up to beta.
        Convention: temperature value is the bilayer tensor one, twice the monolayer
        one.
        """
        if beta is None:
            beta = 4 * self._tau
        if self.verbosity > 0:
            print(f"Launch time evolution for time {beta}")
        if beta < -1e-16:
            raise ValueError("Cannot evolve for negative imaginary time")
        if beta < 3.9 * self._tau:  # care for float round in case beta = 4*tau
            return  # else evolve for 1 step out of niter loop
        niter = round(float(beta / self._tau / 4))  # 2nd order: evolve 2*tau by step

        self.update_bond1(self._gate)
        for i in range(niter - 1):  # there is 1 step out of the loop
            self.update_bond2(self._gate)
            self.update_bond3(self._gate)
            self.update_bond4(self._gate_squared)
            self.update_bond3(self._gate)
            self.update_bond2(self._gate)
            self.update_bond1(self._gate_squared)
            self._beta += 4 * self._tau
        self.update_bond2(self._gate)
        self.update_bond3(self._gate)
        self.update_bond4(self._gate_squared)
        self.update_bond3(self._gate)
        self.update_bond2(self._gate)
        self.update_bond1(self._gate)
        self._beta = round(self._beta + 4 * self._tau, 10)

    def reset_isometries(self):
        if self.verbosity > 0:
            print(f"reset isometries at beta = {self._beta}")
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
                print(f"compute isoA1 and isoB1: eff_rep1 = {self._phys * self._rep1}")
                print(f"aux_rep1 = {self._anc * self._rep2 * self._rep3 * self._rep4}")
            p_data = get_projector_chained(
                self._phys,
                self._anc,
                self._rep1,
                self._rep2,
                self._rep3,
                self._rep4,
                singlet_only=True,
            )
            p_data = p_data.reshape(-1, p_data.shape[6])
            p_transpA = construct_matrix_projector(
                (self._anc, self._rep2, self._rep3, self._rep4),
                (self._phys, self._rep1),
            )
            p_transpA = p_transpA.transpose(4, 0, 5, 1, 2, 3, 6).reshape(p_data.shape)
            self._isoA1 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._rep1, self._phys),
                (self._anc, self._rep2, self._rep3, self._rep4),
            )
            p_transpB = p_transpB.transpose(1, 2, 0, 3, 4, 5, 6).reshape(p_data.shape)
            self._isoB1 = p_transpB.T @ p_data
        return self._isoA1, self._isoB1

    def get_isoAB2(self):
        if self._isoA2 is None:
            if self.verbosity > 1:
                print(f"compute isoA2 and isoB2: eff_rep2 = {self._phys * self._rep2}")
                print(f"aux_rep2 = {self._anc * self._rep1 * self._rep3 * self._rep4}")
            p_data = get_projector_chained(
                self._phys,
                self._anc,
                self._rep1,
                self._rep2,
                self._rep3,
                self._rep4,
                singlet_only=True,
            )
            p_data = p_data.reshape(-1, p_data.shape[6])

            p_transpA = construct_matrix_projector(
                (self._anc, self._rep1, self._rep3, self._rep4),
                (self._phys, self._rep2),
            )
            p_transpA = p_transpA.transpose(4, 0, 1, 5, 2, 3, 6).reshape(p_data.shape)
            self._isoA2 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._rep2, self._phys),
                (self._anc, self._rep1, self._rep3, self._rep4),
            )
            p_transpB = p_transpB.transpose(1, 2, 3, 0, 4, 5, 6).reshape(p_data.shape)
            self._isoB2 = p_transpB.T @ p_data
        return self._isoA2, self._isoB2

    def get_isoAB3(self):
        if self._isoA3 is None:
            if self.verbosity > 1:
                print(f"compute isoA3 and isoB3: eff_rep3 = {self._phys * self._rep3}")
                print(f"aux_rep3 = {self._anc * self._rep1 * self._rep2 * self._rep4}")
            p_data = get_projector_chained(
                self._phys,
                self._anc,
                self._rep1,
                self._rep2,
                self._rep3,
                self._rep4,
                singlet_only=True,
            )
            p_data = p_data.reshape(-1, p_data.shape[6])
            p_transpA = construct_matrix_projector(
                (self._anc, self._rep1, self._rep2, self._rep4),
                (self._phys, self._rep3),
            )
            p_transpA = p_transpA.transpose(4, 0, 1, 2, 5, 3, 6).reshape(p_data.shape)
            self._isoA3 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._rep3, self._phys),
                (self._anc, self._rep1, self._rep2, self._rep4),
            )
            p_transpB = p_transpB.transpose(1, 2, 3, 4, 0, 5, 6).reshape(p_data.shape)
            self._isoB3 = p_transpB.T @ p_data
        return self._isoA3, self._isoB3

    def get_isoAB4(self):
        if self._isoA4 is None:
            if self.verbosity > 1:
                print(f"compute isoA4 and isoB4: eff_rep4 = {self._phys * self._rep4}")
                print(f"aux_rep4 = {self._anc * self._rep1 * self._rep2 * self._rep3}")
            p_data = get_projector_chained(
                self._phys,
                self._anc,
                self._rep1,
                self._rep2,
                self._rep3,
                self._rep4,
                singlet_only=True,
            )
            p_data = p_data.reshape(-1, p_data.shape[6])
            p_transpA = construct_matrix_projector(
                (self._anc, self._rep1, self._rep2, self._rep3),
                (self._phys, self._rep4),
            )
            p_transpA = p_transpA.transpose(4, 0, 1, 2, 3, 5, 6).reshape(p_data.shape)
            self._isoA4 = p_transpA.T @ p_data

            # impose same strucure p-a-1-2-3-4 for A and B
            p_transpB = construct_matrix_projector(
                (self._rep4, self._phys),
                (self._anc, self._rep1, self._rep2, self._rep3),
            )
            p_transpB = p_transpB.transpose(1, 2, 3, 4, 5, 0, 6).reshape(p_data.shape)
            self._isoB4 = p_transpB.T @ p_data
        return self._isoA4, self._isoB4

    def update_bond1(self, gate):
        eff_rep = self._phys * self._rep1
        aux_rep = self._anc * self._rep2 * self._rep3 * self._rep4
        if self.verbosity > 2:
            print(f"update bond 1: rep1 = {self._rep1}, aux_rep = {aux_rep}")

        isoA, isoB = self.get_isoAB1()
        transposedA = isoA @ self._dataA
        matA = SU2_Matrix.from_raw_data(transposedA, aux_rep, eff_rep)
        transposedB = isoB @ self._dataB
        matB = SU2_Matrix.from_raw_data(transposedB, eff_rep, aux_rep)

        newA, self._weights1, newB, new_rep1 = self.update_first_neighbor(
            matA, matB, self._weights1, self._rep1, gate
        )
        if new_rep1 != self._rep1:
            self.reset_isometries()
            self._rep1 = new_rep1
            isoA, isoB = self.get_isoAB1()
        self._dataA = isoA.T @ newA.to_raw_data()
        self._dataB = isoB.T @ newB.to_raw_data()

    def update_bond2(self, gate):
        eff_rep = self._phys * self._rep2
        aux_rep = self._anc * self._rep1 * self._rep3 * self._rep4
        if self.verbosity > 2:
            print(f"update bond 2: rep2 = {self._rep2}, aux_rep = {aux_rep}")

        isoA, isoB = self.get_isoAB2()
        transposedA = isoA @ self._dataA
        matA = SU2_Matrix.from_raw_data(transposedA, aux_rep, eff_rep)
        transposedB = isoB @ self._dataB
        matB = SU2_Matrix.from_raw_data(transposedB, eff_rep, aux_rep)

        newA, self._weights2, newB, new_rep2 = self.update_first_neighbor(
            matA, matB, self._weights2, self._rep2, gate
        )
        if new_rep2 != self._rep2:
            self.reset_isometries()
            self._rep2 = new_rep2
            isoA, isoB = self.get_isoAB2()

        self._dataA = isoA.T @ newA.to_raw_data()
        self._dataB = isoB.T @ newB.to_raw_data()

    def update_bond3(self, gate):
        eff_rep = self._phys * self._rep3
        aux_rep = self._anc * self._rep1 * self._rep2 * self._rep4
        if self.verbosity > 2:
            print(f"update bond 3: rep3 = {self._rep3}, aux_rep = {aux_rep}")

        isoA, isoB = self.get_isoAB3()
        transposedA = isoA @ self._dataA
        matA = SU2_Matrix.from_raw_data(transposedA, aux_rep, eff_rep)
        transposedB = isoB @ self._dataB
        matB = SU2_Matrix.from_raw_data(transposedB, eff_rep, aux_rep)

        newA, self._weights3, newB, new_rep3 = self.update_first_neighbor(
            matA, matB, self._weights3, self._rep3, gate
        )
        if new_rep3 != self._rep3:
            self.reset_isometries()
            self._rep3 = new_rep3
            isoA, isoB = self.get_isoAB3()

        self._dataA = isoA.T @ newA.to_raw_data()
        self._dataB = isoB.T @ newB.to_raw_data()

    def update_bond4(self, gate):
        eff_rep = self._phys * self._rep4
        aux_rep = self._anc * self._rep1 * self._rep2 * self._rep3
        if self.verbosity > 2:
            print(f"update bond 4: rep4 = {self._rep4}, aux_rep = {aux_rep}")

        isoA, isoB = self.get_isoAB4()
        transposedA = isoA @ self._dataA
        matA = SU2_Matrix.from_raw_data(transposedA, aux_rep, eff_rep)
        transposedB = isoB @ self._dataB
        matB = SU2_Matrix.from_raw_data(transposedB, eff_rep, aux_rep)

        newA, self._weights4, newB, new_rep4 = self.update_first_neighbor(
            matA, matB, self._weights4, self._rep4, gate
        )
        if new_rep4 != self._rep4:
            self.reset_isometries()
            self._rep4 = new_rep4
            isoA, isoB = self.get_isoAB4()

        self._dataA = isoA.T @ newA.to_raw_data()
        self._dataB = isoB.T @ newB.to_raw_data()

    def update_first_neighbor(self, matL0, matR0, weights, virt_mid, gate):
        r"""
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
