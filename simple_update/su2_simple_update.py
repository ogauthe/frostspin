import numpy as np

from groups.su2_representation import (
    SU2_Representation,
    SU2_Matrix,
    get_projector_chained,
    construct_matrix_projector,
)


class SU2_SimpleUpdate1x2(object):
    def __init__(
        self, d, Dstar=None, tau=None, h=None, rcutoff=1e-11, file=None, verbosity=0
    ):
        """
        SU(2) symmetric simple update algorithm on plaquette AB. Only deals with finite
        temperature. Each site has to be an *irreducible* representation of SU(2).

        Parameters
        ----------
        d : int
            dimension of physical SU(2) irreducible reprsentation on each site.
        Dstar : int
            Maximal bond dimension, considering only independent multiplets. Not read if
            file is given, retrieved from save.
        tau : float
            Imaginary time step. Not read if file is given.
        h : (d**2, d**2) float or complex ndarray
            Hamltionian. Must be real symmetric or hermitian. Not read if file is given.
        rcutoff : float, optional.
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        file : str, optional
            Save file containing data to restart computation from. File must follow
            save_to_file / load_from_file syntax. If file is provided, Dstar, tau, h and
            cutoff are not read.
        verbosity : int
          Level of log verbosity. Default is no log.


        Conventions for leg ordering
        ----------------------------
              1     3
              |     |
           4--A--2--B--4
              |     |
              3     1
        """
        self.verbosity = verbosity
        if self.verbosity > 0:
            print(f"Construct SU2_SimpleUpdate1x2 with local irrep = {d}")

        if file is not None:  # do not read optional input values, restart from file
            self.load_from_file(file)
            return

        if h.shape != (d ** 2, d ** 2):
            raise ValueError("invalid shape for Hamiltonian")

        if self.verbosity > 0:
            print(f"tau = {tau}, D* = {Dstar}")
            print("Initialize SU2_SimpleUpdate2x2 from beta = 0 thermal product state")

        self._d = d
        self._phys = SU2_Representation.irrep(d)
        self._a = d
        self._anc = self._phys
        self.rcutoff = rcutoff
        self.Dstar = Dstar

        self._h = SU2_Matrix.from_dense(
            h, (self._phys, self._phys), (self._phys, self._phys)
        )
        self.tau = tau

        # only consider thermal equilibrium, start from product state at beta=0
        self._beta = 0.0
        self._rep1 = SU2_Representation.irrep(1)
        self._rep2 = SU2_Representation.irrep(1)
        self._rep3 = SU2_Representation.irrep(1)
        self._rep4 = SU2_Representation.irrep(1)
        self._weights1 = np.ones(1)
        self._weights2 = np.ones(1)
        self._weights3 = np.ones(1)
        self._weights4 = np.ones(1)
        self._dataA = np.ones(1)
        self._dataB = np.ones(1)
        self.reset_isometries()

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

    def load_from_file(self, file):
        if self.verbosity > 0:
            print("Restart SU2_SimpleUpdate1x2 from file", file)
        with np.load(file) as data:
            self._weights1 = data["_SU2_SU1x2_lambda1"]
            self._weights2 = data["_SU2_SU1x2_lambda2"]
            self._weights3 = data["_SU2_SU1x2_lambda3"]
            self._weights4 = data["_SU2_SU1x2_lambda4"]
            self._colors1 = data["_SU2_SU1x2_colors1"]
            self._colors2 = data["_SU2_SU1x2_colors2"]
            self._colors3 = data["_SU2_SU1x2_colors3"]
            self._colors4 = data["_SU2_SU1x2_colors4"]
            self._gammaA = data["_SU2_SU1x2_gammaA"]
            self._gammaB = data["_SU2_SU1x2_gammaB"]
            self._eigvals_h = data["_SU2_SU1x2_eigvals_h"]
            self._eigvecs_h = data["_SU2_SU1x2_eigvecs_h"]
            self.tau = data["_SU2_SU1x2_tau"][()]
            self._beta = data["_SU2_SU1x2_beta"][()]
            self.Dstar = data["_SU2_SU1x2_Dstar"][()]
            self.cutoff = data["_SU2_SU1x2_cutoff"][()]
        self._D1 = self._lambda1.size
        self._D2 = self._lambda2.size
        self._D3 = self._lambda3.size
        self._D4 = self._lambda4.size

    def save_to_file(self, file):
        data = {}
        data["_SU2_SU1x2_lambda1"] = self._lambda1
        data["_SU2_SU1x2_lambda2"] = self._lambda2
        data["_SU2_SU1x2_lambda3"] = self._lambda3
        data["_SU2_SU1x2_lambda4"] = self._lambda4
        data["_SU2_SU1x2_colors1"] = self._colors1
        data["_SU2_SU1x2_colors2"] = self._colors2
        data["_SU2_SU1x2_colors3"] = self._colors3
        data["_SU2_SU1x2_colors4"] = self._colors4
        data["_SU2_SU1x2_gammaA"] = self._gammaA
        data["_SU2_SU1x2_gammaB"] = self._gammaB
        data["_SU2_SU1x2_eigvals_h"] = self._eigvals_h
        data["_SU2_SU1x2_eigvecs_h"] = self._eigvecs_h
        data["_SU2_SU1x2_tau"] = self._tau
        data["_SU2_SU1x2_beta"] = self._beta
        data["_SU2_SU1x2_Dstar"] = self.Dstar
        data["_SU2_SU1x2_cutoff"] = self.cutoff
        np.savez_compressed(file, **data)
        if self.verbosity > 0:
            print("Simple update data stored in file", file)

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
