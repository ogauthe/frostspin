import numpy as np

from su2_representation import SU2_Representation, SU2_Matrix


class SU2_SimpleUpdate1x2(object):
    def __init__(
        self, d, Dstar=None, tau=None, h=None, cutoff=1e-12, file=None, verbosity=0
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
        cutoff : float, optional.
            Singular values smaller than cutoff are set to zero to improve stability.
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
        self._a = d
        self.cutoff = cutoff
        self.Dstar = Dstar

        phys = SU2_Representation([1], [d])
        self._h = SU2_Matrix.from_dense(h, (phys, phys), (phys, phys))
        self.tau = tau

        # only consider thermal equilibrium, start from product state at beta=0
        self._beta = 0.0
        self._rep1 = SU2_Representation([1], [1])
        self._rep2 = SU2_Representation([1], [1])
        self._rep3 = SU2_Representation([1], [1])
        self._rep4 = SU2_Representation([1], [1])
        self._weights1 = np.ones(1)
        self._weights2 = np.ones(1)
        self._weights3 = np.ones(1)
        self._weights4 = np.ones(1)
        self._dataA = np.ones(1)
        self._dataB = np.ones(1)

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
            self.resymmetrize()
        self.update_bond2(self._gate)
        self.update_bond3(self._gate)
        self.update_bond4(self._gate_squared)
        self.update_bond3(self._gate)
        self.update_bond2(self._gate)
        self.update_bond1(self._gate)
        self._beta = round(self._beta + 4 * self._tau, 10)