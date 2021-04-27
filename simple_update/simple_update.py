import numpy as np

from groups.toolsU1 import default_color, combine_colors, eighU1, svdU1
from misc_tools.svd_tools import svd_truncate


def update_first_neighbor(
    matL,
    matR,
    lambda0,
    gate,
    d,
    Dmax,
    col_L,
    col_R,
    col_bond,
    col_d,
    cutoff,
    degen_ratio,
):
    """
    First neighbor simple update algorithm.

    For clarity, a 1D geometry
    =M0_L -- lambda0 -- M_R0=
    is considered in the notations, but the function is direction-agnostic.

    As input, take tensors gamma_X with added weights on all virtual legs as reshaped
    left (M0_L) and right (M0_R) matrices. Compute SVD of those matrices to reduce
    algorithm complexity. Then contract the obtained tensors into matrix theta, apply
    gate. Split and truncate update matrix to obtain new set of diagonalized weights and
    renormalized tensors.
    """
    # 1) SVD cut between constant tensors and effective tensor to update, hence reduce
    # main SVD to dimension D*d < a*D**3
    #     \|        \|
    #     -L-    -> -cstL==effL-
    #      |\        |        \
    D = lambda0.size
    cstL, svL, effL, col_effL = svdU1(matL, col_L, -combine_colors(col_d, col_bond))
    D_effL = svL.size
    effL *= svL[:, None]
    #     \|                \|
    #     -R-    -> -effR==cstR-
    #      |\        /       |
    effR, svR, cstR, col_effR = svdU1(matR, combine_colors(-col_bond, -col_d), col_R)
    D_effR = svR.size
    effR *= svR

    # 2) construct matrix theta with gate g
    #
    #             =effL-l-effR=
    #                 \  /
    #   theta =        gg
    #                 /  \
    theta = effL.reshape(D_effL * d, D) / lambda0  # remove double counting
    theta = (theta @ effR.reshape(D, d * D_effR)).reshape(D_effL, d, d, D_effR)
    theta = theta.transpose(0, 3, 1, 2).reshape(D_effL * D_effR, d ** 2)
    theta = (theta @ gate).reshape(D_effL, D_effR, d, d)

    # 3) cut theta with SVD
    theta = theta.swapaxes(1, 2).reshape(D_effL * d, D_effR * d)
    U, new_lambda, V, new_col_lambda = svd_truncate(
        theta,
        Dmax,
        row_colors=combine_colors(col_effL, col_d),
        col_colors=combine_colors(col_effR, col_d),
        full=True,
        cutoff=cutoff,
        degen_ratio=degen_ratio,
    )

    # 4) renormalize link dimension
    D = new_lambda.size  # D < Dmax at the begining
    new_lambda /= new_lambda.sum()  # singular values are positive

    # 5) start reconstruction of new gammaX and gammaY by unifying cst and eff parts
    new_effL = (U * new_lambda).reshape(D_effL, d * D)
    newL = cstL @ new_effL
    new_effR = (
        (V * new_lambda[:, None])
        .reshape(D, D_effR, d)
        .swapaxes(1, 2)
        .reshape(D * d, D_effR)
    )
    newR = new_effR @ cstR
    return newL, new_lambda, newR, -new_col_lambda


def update_second_neighbor(
    matL,
    mat_mid,
    matR,
    lambda_L,
    lambda_R,
    gate,
    d,
    Dmax,
    col_L,
    col_mid,
    col_R,
    col_bL,
    col_bR,
    col_d,
    cutoff,
    degen_ratio,
):
    """
    Second and third neighbor simple update algorithm.
    Construct matrix theta from matrices M0_L, M0_mid and M0_R obtained by adding
    diagonal weights to all virtual bonds and reshaping to matrices initial tensors
    gammaX, gammaY and gammaZ.

    For clarity, a 1D geometry
    M_left -- lambda_left -- M_mid -- lambda_right -- M_right
    is considered in the notations, but the function is direction-agnostic and works for
    any geometry with two edge tensors linked by a proxy one.
    """

    D_L, D_R = lambda_L.size, lambda_R.size

    # 1) SVD cut between constant tensors and effective tensors to update
    #     \|        \|
    #     -L-    -> -cstL==effL-lambda_L- (D_L)
    #      |\        |       \
    cstL, svL, effL, col_effL = svdU1(matL, col_L, -combine_colors(col_d, col_bL))
    D_effL = svL.size
    effL = (svL[:, None] * effL).reshape(D_effL * d, D_L) / lambda_L  # remove lambda
    #                       \|/|
    #                       cstM
    #     \|                 ||
    #     -M-   ->  (D_L) - effM - (D_R)
    #      |\
    eff_m, sv_m, cst_m, col_effm = svdU1(
        mat_mid, -combine_colors(col_bL, col_bR), col_mid
    )
    D_effm = sv_m.size
    eff_m = (eff_m * sv_m).reshape(D_L, D_R * D_effm)
    #     \|                              \|
    #     -R-   ->  (D_R)  lambda_R-effR==cstR
    #      |\                              |\
    effR, svR, cstR, col_effR = svdU1(matR, -combine_colors(col_bR, col_d), col_R)
    D_effR = svR.size
    effR = (effR * svR).reshape(D_R, d * D_effR) / lambda_R[:, None]

    # contract tensor network
    #                         ||
    #    =effL-lambdaL -- eff_mid -- lambdaR-effR=
    #         \                             /
    #          \----------- gate ----------/
    theta = (effL @ eff_m).reshape(D_effL * d, D_R, D_effm)
    theta = theta.swapaxes(1, 2).reshape(D_effL * d * D_effm, D_R)
    theta = (theta @ effR).reshape(D_effL, d, D_effm, d, D_effR)
    theta = theta.transpose(0, 2, 4, 1, 3).reshape(D_effL * D_effm * D_effR, d ** 2)
    theta = (theta @ gate).reshape(D_effL, D_effm, D_effR, d, d)
    theta = theta.transpose(0, 3, 1, 2, 4).reshape(D_effL * d, D_effm * D_effR * d)

    # first SVD: cut left part
    U, new_lambda_L, V, col_nbL = svd_truncate(
        theta,
        Dmax,
        row_colors=combine_colors(col_effL, col_d),
        col_colors=-combine_colors(-col_effm, col_effR, col_d),
        full=True,
        cutoff=cutoff,
        degen_ratio=degen_ratio,
    )
    D_L = new_lambda_L.size
    new_lambda_L /= new_lambda_L.sum()
    new_effL = (U * new_lambda_L).reshape(D_effL, d * D_L)

    # second SVD: split middle and right parts
    theta = (new_lambda_L[:, None] * V).reshape(D_L * D_effm, D_effR * d)
    U, new_lambda_R, V, col_nbR = svd_truncate(
        theta,
        Dmax,
        row_colors=combine_colors(col_nbL, -col_effm),
        col_colors=-combine_colors(col_effR, col_d),
        full=True,
        cutoff=cutoff,
        degen_ratio=degen_ratio,
    )
    D_R = new_lambda_R.size
    new_lambda_R /= new_lambda_R.sum()
    new_effm = U.reshape(D_L, D_effm, D_R) * new_lambda_R
    new_effR = V.reshape(D_R, D_effR, d) * new_lambda_R[:, None, None]

    # bring back constant parts
    newL = cstL @ new_effL
    new_effm = new_effm.swapaxes(1, 2).reshape(D_L * D_R, D_effm)
    new_mid = new_effm @ cst_m
    new_effR = new_effR.swapaxes(1, 2).reshape(D_R * d, D_effR)
    newR = new_effR @ cstR

    return newL, new_mid, newR, new_lambda_L, new_lambda_R, -col_nbL, col_nbR


class SimpleUpdate1x2(object):
    # Instead of adding diagonal weights on all bonds before applying gate, it is more
    # efficient to consider weights always on by default. Then weights must be removed
    # only once on single updated link to avoid double counting.
    def __init__(
        self,
        d,
        a,
        Dmax=None,
        tau=None,
        h=None,
        tensors=None,
        colors=None,
        cutoff=1e-10,
        degen_ratio=1.0,
        file=None,
        verbosity=0,
    ):
        """
        Simple update algorithm on plaquette AB.

        Parameters
        ----------
        d : integer
            Dimension of physical leg.
        a : integer
            Dimension of ancilla leg. a=1 for a pure wavefunction and a=d for a thermal
            ensemble.
        Dmax : int
            Maximal bond dimension. If provided, tensors may have different D at
            initialization. Not read if file is given, retrieved from save.
        tau : float
            Imaginary time step. Not read if file is given.
        h : (d**2, d**2) float or complex ndarray
            Hamltionian. Must be real symmetric or hermitian. Not read if file is given.
        tensors : optional, enumerable of 4 ndarrays with shapes (d,a,D,D,D,D)
            Initial tensors. If not provided, random tensors are taken. Not read if file
            is given.
        colors : optional, if provided either integer array of dimension d or enumerable
            of 6 integer arrays for physical, ancilla + 4 virtual legs matching tensors.
            Quantum numbers for physical leg / physical, ancilla and virtual legs. Not
            read if file is given. If not provided at first construction, no symmetry is
            assumed.
        cutoff : float, optional.
            Singular values smaller than cutoff are set to zero to improve stability.
            Default is 1e-10.
        degen_ratio : float, optional.
            Used to define multiplets in singular values and truncate between two
            multiplets. Two consecutive singular values are considered degenerate if
            1 >= s[i+1]/s[i] >= degen_ratio > 0. Default is 1.0 (exact degeneracies)
        file : str, optional
            Save file containing data to restart computation from. File must follow
            save_to_file / load_from_file syntax. If file is provided, d and a are read
            to check consistency between save and input, the other parameter (except
            verbosity) are not read and directly set from file.
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

        self._d = d
        self._a = a
        self.verbosity = verbosity
        if self.verbosity > 0:
            print(f"construct SimpleUpdate1x2 with d = {d}, a = {a} and Dmax = {Dmax}")

        if file is not None:  # do not read optional input values, restart from file
            self.load_from_file(file)
            return

        self.cutoff = cutoff
        self.degen_ratio = degen_ratio
        self.Dmax = Dmax
        if h.shape != (d ** 2, d ** 2):
            raise ValueError("invalid shape for Hamiltonian")

        if colors is None:  # default color whatever tensor shapes
            self._colors_p = default_color
            self._colors_a = default_color
            self._colors1 = default_color
            self._colors2 = default_color
            self._colors3 = default_color
            self._colors4 = default_color

        # consider 3 cases:
        # 1) tensors are provided: check dimension match and nothing else
        # 2) tensors are not provided and a == 1: pure state, start from random tensors
        # 3) tensors are not provided and a == d: thermal equilibrium, start from beta=0

        # 1) provided tensors
        if tensors is not None:
            if self.verbosity > 0:
                print("Initialize SimpleUpdate2x2 from given tensors")

            A0, B0 = tensors
            self._D1, self._D2, self._D3, self._D4 = A0.shape[2:]
            if A0.shape != (d, a, self._D1, self._D2, self._D3, self._D4):
                raise ValueError("invalid shape for A0")
            if B0.shape != (d, a, self._D3, self._D4, self._D1, self._D2):
                raise ValueError("invalid shape for B0")

            if colors is not None:
                if len(colors) != 6:
                    raise ValueError(
                        "With given initial tensors, colors must be",
                        "[colors_p, colors_a, colors_1...6]",
                    )
                if len(colors[0]) != d:
                    raise ValueError("physical leg colors length is not d")
                self._colors_p = np.asarray(colors[0], dtype=np.int8)
                if len(colors[1]) != a:
                    raise ValueError("ancilla leg colors length is not a")
                self._colors_a = np.asarray(colors[1], dtype=np.int8)
                if len(colors[2]) != self._D1:
                    raise ValueError("virtual leg 1 colors length is not D1")
                self._colors1 = np.asarray(colors[2], dtype=np.int8)
                if len(colors[3]) != self._D2:
                    raise ValueError("virtual leg 2 colors length is not D2")
                self._colors2 = np.asarray(colors[3], dtype=np.int8)
                if len(colors[4]) != self._D3:
                    raise ValueError("virtual leg 3 colors length is not D3")
                self._colors3 = np.asarray(colors[4], dtype=np.int8)
                if len(colors[5]) != self._D4:
                    raise ValueError("virtual leg 4 colors length is not D4")

        # 2) pure state
        elif a == 1:
            if self.verbosity > 0:
                print("Initialize SimpleUpdate2x2 from random pure state")
            if colors is not None:  # can fix, easy to generate random U(1) tensors
                raise ValueError(
                    "Initial tensors must be provided to use colors in pure state"
                )
            self._D1 = Dmax
            self._D2 = Dmax
            self._D3 = Dmax
            self._D4 = Dmax
            A0 = np.random.random((d, 1, self._D1, self._D2, self._D3, self._D4)) - 0.5
            B0 = np.random.random((d, 1, self._D3, self._D4, self._D1, self._D2)) - 0.5
            self._gammaA = A0 / np.amax(A0)
            self._gammaB = B0 / np.amax(B0)

        # 3) thermal equilibrium, start from product state at beta=0
        elif a == d:
            if self.verbosity > 0:
                print("Initialize SimpleUpdate2x2 from beta=0 thermal product state")
            self._D1 = 1
            self._D2 = 1
            self._D3 = 1
            self._D4 = 1
            self._gammaA = np.eye(d).reshape(d, a, 1, 1, 1, 1)
            self._gammaB = np.eye(d).reshape(d, a, 1, 1, 1, 1)
            if colors is not None:
                if len(colors) != d:
                    raise ValueError(
                        "For beta=0 thermal equilibrium, colors must be colors_p"
                    )
                self._colors_p = np.ascontiguousarray(colors, dtype=np.int8)
                self._colors_a = -self._colors_p
                self._colors1 = np.zeros(1, dtype=np.int8)
                self._colors2 = np.zeros(1, dtype=np.int8)
                self._colors3 = np.zeros(1, dtype=np.int8)
                self._colors4 = np.zeros(1, dtype=np.int8)

        else:
            raise ValueError("If tensors are not provided, a must be 1 or d")

        # wait for colors_p to be set to use U(1) in h1 and h2 diagonalization.
        colors_h = combine_colors(self._colors_p, -self._colors_p)
        self._eigvals_h, self._eigvecs_h, _ = eighU1(h, colors_h)
        self.tau = tau  # need eigvals and eigvecs to set tau
        self._beta = 0.0

        # now that dimensions are known, initialize weights to 1.
        self._lambda1 = np.ones(self._D1)
        self._lambda2 = np.ones(self._D2)
        self._lambda3 = np.ones(self._D3)
        self._lambda4 = np.ones(self._D4)

    @property
    def beta(self):
        return self._beta

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self.verbosity > 0:
            print(f"set tau to {tau}")
        self._tau = tau
        # symmetric / hermitian diagonalization is faster than expm and ensures U(1)
        self._gate = (
            self._eigvecs_h * np.exp(-tau * self._eigvals_h) @ self._eigvecs_h.T.conj()
        )
        self._gate_squared = (
            self._eigvecs_h
            * np.exp(-2 * tau * self._eigvals_h)
            @ self._eigvecs_h.T.conj()
        )

    @property
    def d(self):
        return self._d

    @property
    def a(self):
        return self._a

    @property
    def Ds(self):
        return (self._D1, self._D2, self._D3, self._D4)

    @property
    def h(self):
        return self._eigvecs_h * self._eigvals_h @ self._eigvecs_h.T.conj()

    @property
    def colors(self):
        """
        Tuple
        U(1) quantum numbers for non-equivalent legs.
        Convention: sort as ((physical,ancilla),leg_i) to have colors[i] = color_i.
        """
        return (
            (self._colors_p, self._colors_a),
            self._colors1,
            self._colors2,
            self._colors3,
            self._colors4,
        )

    @property
    def lambdas(self):
        """
        Tuple
        Simple update weights.
        Convention: return ((None,None),leg_i) to be consistent with colors.
        """
        return (
            (None, None),
            self._lambda1,
            self._lambda2,
            self._lambda3,
            self._lambda4,
        )

    def load_from_file(self, file):
        if self.verbosity > 0:
            print("Restart simple update from file", file)
        # do not read tau tand Dmax, set them from __init__ input
        with np.load(file) as data:
            self._lambda1 = data["_SU1x2_lambda1"]
            self._lambda2 = data["_SU1x2_lambda2"]
            self._lambda3 = data["_SU1x2_lambda3"]
            self._lambda4 = data["_SU1x2_lambda4"]
            self._colors_p = data["_SU1x2_colors_p"]
            self._colors_a = data["_SU1x2_colors_a"]
            self._colors1 = data["_SU1x2_colors1"]
            self._colors2 = data["_SU1x2_colors2"]
            self._colors3 = data["_SU1x2_colors3"]
            self._colors4 = data["_SU1x2_colors4"]
            self._gammaA = data["_SU1x2_gammaA"]
            self._gammaB = data["_SU1x2_gammaB"]
            self._eigvals_h = data["_SU1x2_eigvals_h"]
            self._eigvecs_h = data["_SU1x2_eigvecs_h"]
            self.tau = data["_SU1x2_tau"][()]
            self._beta = data["_SU1x2_beta"][()]
            self.Dmax = data["_SU1x2_Dmax"][()]
            if "_SU1x2_cutoff" in data.files:
                self.cutoff = data["_SU1x2_cutoff"][()]
            else:
                self.cutoff = 1e-13  # default value for backward compatibility
            if "_SU1x2_degen_ratio" in data.files:
                self.degen_ratio = data["_SU1x2_degen_ratio"][()]
            else:
                self.degen_ratio = None
        self._D1 = self._lambda1.size
        self._D2 = self._lambda2.size
        self._D3 = self._lambda3.size
        self._D4 = self._lambda4.size

        if self._d != self._gammaA.shape[0]:
            raise ValueError("Physical dimension differs from save")
        if self._a != self._gammaA.shape[1]:
            raise ValueError("Ancila dimension differs from save")

    def save_to_file(self, file=None):
        data = {}
        data["_SU1x2_lambda1"] = self._lambda1
        data["_SU1x2_lambda2"] = self._lambda2
        data["_SU1x2_lambda3"] = self._lambda3
        data["_SU1x2_lambda4"] = self._lambda4
        data["_SU1x2_colors_p"] = self._colors_p
        data["_SU1x2_colors_a"] = self._colors_a
        data["_SU1x2_colors1"] = self._colors1
        data["_SU1x2_colors2"] = self._colors2
        data["_SU1x2_colors3"] = self._colors3
        data["_SU1x2_colors4"] = self._colors4
        data["_SU1x2_gammaA"] = self._gammaA
        data["_SU1x2_gammaB"] = self._gammaB
        data["_SU1x2_eigvals_h"] = self._eigvals_h
        data["_SU1x2_eigvecs_h"] = self._eigvecs_h
        data["_SU1x2_tau"] = self._tau
        data["_SU1x2_beta"] = self._beta
        data["_SU1x2_Dmax"] = self.Dmax
        data["_SU1x2_cutoff"] = self.cutoff
        if self.degen_ratio is not None:  # do not save if None (avoid dtype=object)
            data["_SU1x2_degen_ratio"] = self.degen_ratio
        if file is None:
            return data
        np.savez_compressed(file, **data)
        if self.verbosity > 0:
            print("Simple update data stored in file", file)

    def get_AB(self):
        """
        Return optimized tensors A and B.
        Tensors are obtained by adding relevant sqrt(lambda) to every leg of gammaX
        """
        # actually weights are on by default, so *remove* sqrt(lambda)
        sl1 = 1 / np.sqrt(self._lambda1)
        sl2 = 1 / np.sqrt(self._lambda2)
        sl3 = 1 / np.sqrt(self._lambda3)
        sl4 = 1 / np.sqrt(self._lambda4)
        A = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaA, sl1, sl2, sl3, sl4)
        B = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaB, sl3, sl4, sl1, sl2)
        A /= np.amax(A)
        B /= np.amax(B)
        return A, B

    def get_colors_AB(self):
        """
        Return colors of optimized tensors A and B.
        """
        colorsA = (
            self._colors_p,
            self._colors_a,
            self._colors1,
            self._colors2,
            self._colors3,
            self._colors4,
        )
        colorsB = (
            -self._colors_p,
            -self._colors_a,
            -self._colors3,
            -self._colors4,
            -self._colors1,
            -self._colors2,
        )
        return colorsA, colorsB

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
        self.update_bond2(self._gate)
        self.update_bond3(self._gate)
        self.update_bond4(self._gate_squared)
        self.update_bond3(self._gate)
        self.update_bond2(self._gate)
        self.update_bond1(self._gate)
        self._beta = round(self._beta + 4 * niter * self._tau, 10)

    def bond_entanglement_entropy(self):
        """
        Compute the entanglement entropy on every bonds as S = -sum_i p_i log_pi
        """
        s = np.empty(4)
        s[0] = -self._lambda1 @ np.log(self._lambda1)
        s[1] = -self._lambda2 @ np.log(self._lambda2)
        s[2] = -self._lambda3 @ np.log(self._lambda3)
        s[3] = -self._lambda4 @ np.log(self._lambda4)
        return s

    def update_bond1(self, gate):
        # add diagonal weights to gammaA and gammaC
        M_A = self._gammaA.transpose(1, 3, 4, 5, 0, 2).reshape(
            self._a * self._D2 * self._D3 * self._D4, self._d * self._D1
        )
        M_B = self._gammaB.transpose(4, 0, 1, 2, 3, 5).reshape(
            self._D1 * self._d, self._a * self._D3 * self._D4 * self._D2
        )

        col_L = combine_colors(
            self._colors_a, self._colors2, self._colors3, self._colors4
        )
        col_R = combine_colors(
            self._colors_a, self._colors3, self._colors4, self._colors2
        )
        # construct matrix theta, renormalize bond dimension and get back tensors
        M_A, self._lambda1, M_B, self._colors1 = update_first_neighbor(
            M_A,
            M_B,
            self._lambda1,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors1,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D1 = self._lambda1.size
        # define new gammaA and gammaC from renormalized M_A and M_C
        self._gammaA = M_A.reshape(
            self._a, self._D2, self._D3, self._D4, self._d, self._D1
        ).transpose(4, 0, 5, 1, 2, 3)
        self._gammaB = M_B.reshape(
            self._D1, self._d, self._a, self._D3, self._D4, self._D2
        ).transpose(1, 2, 3, 4, 0, 5)

        if self.verbosity > 1:
            print("updated bond 1: new lambda1 =", self._lambda1)

    def update_bond2(self, gate):
        M_A = self._gammaA.transpose(1, 2, 4, 5, 0, 3).reshape(
            self._a * self._D1 * self._D3 * self._D4, self._d * self._D2
        )
        M_B = self._gammaB.transpose(5, 0, 1, 2, 3, 4).reshape(
            self._D2 * self._d, self._a * self._D3 * self._D4 * self._D1
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors3, self._colors4
        )
        col_R = combine_colors(
            self._colors_a, self._colors3, self._colors4, self._colors1
        )
        M_A, self._lambda2, M_B, self._colors2 = update_first_neighbor(
            M_A,
            M_B,
            self._lambda2,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors2,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D2 = self._lambda2.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D3, self._D4, self._d, self._D2
        ).transpose(4, 0, 1, 5, 2, 3)
        self._gammaB = M_B.reshape(
            self._D2, self._d, self._a, self._D3, self._D4, self._D1
        ).transpose(1, 2, 3, 4, 5, 0)

        if self.verbosity > 1:
            print("updated bond 2: new lambda2 =", self._lambda2)

    def update_bond3(self, gate):
        M_A = self._gammaA.transpose(1, 2, 3, 5, 0, 4).reshape(
            self._a * self._D1 * self._D2 * self._D4, self._d * self._D3
        )
        M_B = self._gammaB.transpose(2, 0, 1, 3, 4, 5).reshape(
            self._D3 * self._d, self._a * self._D4 * self._D1 * self._D2
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors4
        )
        col_R = combine_colors(
            self._colors_a, self._colors4, self._colors1, self._colors2
        )
        M_A, self._lambda3, M_B, self._colors3 = update_first_neighbor(
            M_A,
            M_B,
            self._lambda3,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors3,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D3 = self._lambda3.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D4, self._d, self._D3
        ).transpose(4, 0, 1, 2, 5, 3)
        self._gammaB = M_B.reshape(
            self._D3, self._d, self._a, self._D4, self._D1, self._D2
        ).transpose(1, 2, 0, 3, 4, 5)

        if self.verbosity > 1:
            print("updated bond 3: new lambda3 =", self._lambda3)

    def update_bond4(self, gate):
        M_A = self._gammaA.transpose(1, 2, 3, 4, 0, 5).reshape(
            self._a * self._D1 * self._D2 * self._D3, self._d * self._D4
        )
        M_B = self._gammaB.transpose(3, 0, 1, 2, 4, 5).reshape(
            self._D4 * self._d, self._a * self._D3 * self._D1 * self._D2
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors3
        )
        col_R = combine_colors(
            self._colors_a, self._colors3, self._colors1, self._colors2
        )
        M_A, self._lambda4, M_B, self._colors4 = update_first_neighbor(
            M_A,
            M_B,
            self._lambda4,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors4,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D4 = self._lambda4.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D3, self._d, self._D4
        ).transpose(4, 0, 1, 2, 3, 5)
        self._gammaB = M_B.reshape(
            self._D4, self._d, self._a, self._D3, self._D1, self._D2
        ).transpose(1, 2, 3, 0, 4, 5)

        if self.verbosity > 1:
            print("updated bond 4: new lambda4 =", self._lambda4)


class SimpleUpdate2x2(object):
    # Same as SimpleUpdate1x2, consider weights on by default.
    def __init__(
        self,
        d,
        a,
        Dmax=None,
        tau=None,
        h1=None,
        h2=None,
        tensors=None,
        colors=None,
        cutoff=1e-10,
        degen_ratio=1.0,
        file=None,
        verbosity=0,
    ):
        """
        Simple update algorithm on plaquette AB//CD.

        Parameters
        ----------
        d : integer
          Dimension of physical leg.
        a : integer
          Dimension of ancilla leg. a=1 for a pure wavefunction and a=d for a thermal
          ensemble.
        Dmax : int
          Maximal bond dimension. If provided, tensors may have different D at
          initialization. Not read if file is given, retrieved from save.
        tau : float
          Imaginary time step. Not read if file is given.
        h1 : (d**2, d**2) float or complex ndarray
          First neigbor Hamltionian. Must be real symmetric or hermitian. Not read if
          file is given.
        h2 : (d**2, d**2) float or complex ndarray
          Second neigbor Hamltionian. Must be real symmetric or hermitian. Not read if
          file is given.
        tensors : optional, enumerable of 4 ndarrays with shapes (d,a,D,D,D,D)
          Initial tensors. If not provided, random tensors are taken. Not read if file
          is given.
        colors : optional, if provided either integer array of dimension d or enumerable
          of 10 integer arrays for physical, ancilla + 8 virtual legs matching tensors.
          Quantum numbers for physical leg / physical, ancilla and virtual legs. Not
          read if file is given. If not provided at first construction, no symmetry is
          assumed.
        cutoff : float, optional.
            Singular values smaller than cutoff are set to zero to improve stability.
            Default is 1e-10.
        degen_ratio : float, optional.
            Used to define multiplets in singular values and truncate between two
            multiplets. Two consecutive singular values are considered degenerate if
            1 >= s[i+1]/s[i] >= degen_ratio > 0. Default is 1.0 (exact degeneracies)
        file : str, optional
          Save file containing data to restart computation from. File must follow
          save_to_file / load_from_file syntax. If file is provided, d and a are read to
          check consistency between save and input, the other parameter (except
          verbosity) are not read and directly set from file.
        verbosity : int
          Level of log verbosity. Default is no log.


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
        """

        self._d = d
        self._a = a
        self.verbosity = verbosity
        if self.verbosity > 0:
            print(f"construct SimpleUpdate2x2 with d = {d}, a = {a} and Dmax = {Dmax}")

        if file is not None:  # do not read optional input values, restart from file
            self.load_from_file(file)
            return

        self.cutoff = cutoff
        self.degen_ratio = degen_ratio
        self.Dmax = Dmax
        if h1.shape != (d ** 2, d ** 2):
            raise ValueError("invalid shape for Hamiltonian h1")
        if h2.shape != (d ** 2, d ** 2):
            raise ValueError("invalid shape for Hamiltonian h2")

        if colors is None:  # default color whatever tensor shapes
            self._colors_p = default_color
            self._colors_a = default_color
            self._colors1 = default_color
            self._colors2 = default_color
            self._colors3 = default_color
            self._colors4 = default_color
            self._colors5 = default_color
            self._colors6 = default_color
            self._colors7 = default_color
            self._colors8 = default_color

        # consider 3 cases:
        # 1) tensors are provided: check dimension match and nothing else
        # 2) tensors are not provided and a == 1: pure state, start from random tensors
        # 3) tensors are not provided and a == d: thermal equilibrium, start from beta=0

        # 1) provided tensors
        if tensors is not None:
            if self.verbosity > 0:
                print("Initialize SimpleUpdate2x2 from given tensors")

            A0, B0, C0, D0 = tensors
            self._D1, self._D2, self._D3, self._D4 = A0.shape[2:]
            self._D6, self._D8, self._D5, self._D7 = D0.shape[2:]
            if A0.shape != (d, a, self._D1, self._D2, self._D3, self._D4):
                raise ValueError("invalid shape for A0")
            if B0.shape != (d, a, self._D5, self._D4, self._D6, self._D2):
                raise ValueError("invalid shape for B0")
            if C0.shape != (d, a, self._D3, self._D7, self._D1, self._D8):
                raise ValueError("invalid shape for C0")
            if D0.shape != (d, a, self._D6, self._D8, self._D5, self._D7):
                raise ValueError("invalid shape for D0")

            if colors is not None:
                if len(colors) != 10:
                    raise ValueError(
                        "With given initial tensors, colors must be",
                        "[colors_p, colors_a, colors_1...8]",
                    )
                if len(colors[0]) != d:
                    raise ValueError("physical leg colors length is not d")
                self._colors_p = np.asarray(colors[0], dtype=np.int8)
                if len(colors[1]) != a:
                    raise ValueError("ancilla leg colors length is not a")
                self._colors_a = np.asarray(colors[1], dtype=np.int8)
                if len(colors[2]) != self._D1:
                    raise ValueError("virtual leg 1 colors length is not D1")
                self._colors1 = np.asarray(colors[2], dtype=np.int8)
                if len(colors[3]) != self._D2:
                    raise ValueError("virtual leg 2 colors length is not D2")
                self._colors2 = np.asarray(colors[3], dtype=np.int8)
                if len(colors[4]) != self._D3:
                    raise ValueError("virtual leg 3 colors length is not D3")
                self._colors3 = np.asarray(colors[4], dtype=np.int8)
                if len(colors[5]) != self._D4:
                    raise ValueError("virtual leg 4 colors length is not D4")
                self._colors4 = np.asarray(colors[5], dtype=np.int8)
                if len(colors[6]) != self._D5:
                    raise ValueError("virtual leg 5 colors length is not D5")
                self._colors5 = np.asarray(colors[6], dtype=np.int8)
                if len(colors[7]) != self._D6:
                    raise ValueError("virtual leg 6 colors length is not D6")
                self._colors6 = np.asarray(colors[7], dtype=np.int8)
                if len(colors[8]) != self._D7:
                    raise ValueError("virtual leg 7 colors length is not D7")
                self._colors7 = np.asarray(colors[8], dtype=np.int8)
                if len(colors[9]) != self._D8:
                    raise ValueError("virtual leg 8 colors length is not D8")
                self._colors8 = np.asarray(colors[9], dtype=np.int8)

        # 2) pure state
        elif a == 1:
            if self.verbosity > 0:
                print("Initialize SimpleUpdate2x2 from random pure state")
            if colors is not None:  # can fix, easy to generate random U(1) tensors
                raise ValueError(
                    "Initial tensors must be provided to use colors in pure state"
                )
            self._D1 = Dmax
            self._D2 = Dmax
            self._D3 = Dmax
            self._D4 = Dmax
            self._D5 = Dmax
            self._D6 = Dmax
            self._D7 = Dmax
            self._D8 = Dmax
            A0 = np.random.random((d, 1, self._D1, self._D2, self._D3, self._D4)) - 0.5
            B0 = np.random.random((d, 1, self._D5, self._D4, self._D6, self._D2)) - 0.5
            C0 = np.random.random((d, 1, self._D3, self._D7, self._D1, self._D8)) - 0.5
            D0 = np.random.random((d, 1, self._D6, self._D8, self._D5, self._D7)) - 0.5
            self._gammaA = A0 / np.amax(A0)
            self._gammaB = B0 / np.amax(B0)
            self._gammaC = C0 / np.amax(C0)
            self._gammaD = D0 / np.amax(D0)

        # 3) thermal equilibrium, start from product state at beta=0
        elif a == d:
            if self.verbosity > 0:
                print("Initialize SimpleUpdate2x2 from beta=0 thermal product state")
            self._D1 = 1
            self._D2 = 1
            self._D3 = 1
            self._D4 = 1
            self._D5 = 1
            self._D6 = 1
            self._D7 = 1
            self._D8 = 1
            self._gammaA = np.eye(d).reshape(d, a, 1, 1, 1, 1)
            self._gammaB = np.eye(d).reshape(d, a, 1, 1, 1, 1)
            self._gammaC = np.eye(d).reshape(d, a, 1, 1, 1, 1)
            self._gammaD = np.eye(d).reshape(d, a, 1, 1, 1, 1)
            if colors is not None:
                if len(colors) != d:
                    raise ValueError(
                        "For beta=0 thermal equilibrium, colors must be colors_p"
                    )
                self._colors_p = np.ascontiguousarray(colors, dtype=np.int8)
                self._colors_a = -self._colors_p
                self._colors1 = np.zeros(1, dtype=np.int8)
                self._colors2 = np.zeros(1, dtype=np.int8)
                self._colors3 = np.zeros(1, dtype=np.int8)
                self._colors4 = np.zeros(1, dtype=np.int8)
                self._colors5 = np.zeros(1, dtype=np.int8)
                self._colors6 = np.zeros(1, dtype=np.int8)
                self._colors7 = np.zeros(1, dtype=np.int8)
                self._colors8 = np.zeros(1, dtype=np.int8)

        else:
            raise ValueError("If tensors are not provided, a must be 1 or d")

        # wait for colors_p to be set to use U(1) in h1 and h2 diagonalization.
        colors_h1 = combine_colors(self._colors_p, -self._colors_p)
        self._eigvals_h1, self._eigvecs_h1, _ = eighU1(h1, colors_h1)
        colors_h2 = combine_colors(self._colors_p, self._colors_p)
        self._eigvals_h2, self._eigvecs_h2, _ = eighU1(h2, colors_h2)
        self.tau = tau  # need eigvals and eigvecs to set tau
        self._beta = 0.0

        # now that dimensions are known, initialize weights to 1.
        self._lambda1 = np.ones(self._D1)
        self._lambda2 = np.ones(self._D2)
        self._lambda3 = np.ones(self._D3)
        self._lambda4 = np.ones(self._D4)
        self._lambda5 = np.ones(self._D5)
        self._lambda6 = np.ones(self._D6)
        self._lambda7 = np.ones(self._D7)
        self._lambda8 = np.ones(self._D8)

    @property
    def beta(self):
        return self._beta

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self.verbosity > 0:
            print(f"set tau to {tau}")
        self._tau = tau
        # symmetric / hermitian diagonalization is faster than expm and ensures U(1)
        self._g1 = (
            self._eigvecs_h1
            * np.exp(-tau * self._eigvals_h1)
            @ self._eigvecs_h1.T.conj()
        )
        self._g1_squared = (
            self._eigvecs_h1
            * np.exp(-2 * tau * self._eigvals_h1)
            @ self._eigvecs_h1.T.conj()
        )
        self._g2_sqrt = (
            self._eigvecs_h2
            * np.exp(-tau / 2 * self._eigvals_h2)
            @ self._eigvecs_h2.T.conj()
        )
        self._g2 = (
            self._eigvecs_h2
            * np.exp(-tau * self._eigvals_h2)
            @ self._eigvecs_h2.T.conj()
        )

    @property
    def d(self):
        return self._d

    @property
    def a(self):
        return self._a

    @property
    def Ds(self):
        return (
            self._D1,
            self._D2,
            self._D3,
            self._D4,
            self._D5,
            self._D6,
            self._D7,
            self._D8,
        )

    @property
    def h1(self):
        return self._eigvecs_h1 * self._eigvals_h1 @ self._eigvecs_h1.T.conj()

    @property
    def h2(self):
        return self._eigvecs_h2 * self._eigvals_h2 @ self._eigvecs_h2.T.conj()

    @property
    def colors(self):
        """
        Tuple
        U(1) quantum numbers for non-equivalent legs.
        Convention: sort as ((physical,ancilla),leg_i) to have colors[i] = color_i.
        """
        return (
            (self._colors_p, self._colors_a),
            self._colors1,
            self._colors2,
            self._colors3,
            self._colors4,
            self._colors5,
            self._colors6,
            self._colors7,
            self._colors8,
        )

    @property
    def lambdas(self):
        """
        Tuple
        Simple update weights.
        Convention: return ((None,None),leg_i) to be consistent with colors.
        """
        return (
            (None, None),
            self._lambda1,
            self._lambda2,
            self._lambda3,
            self._lambda4,
            self._lambda5,
            self._lambda6,
            self._lambda7,
            self._lambda8,
        )

    def load_from_file(self, file):
        if self.verbosity > 0:
            print("Restart simple update from file", file)
        # do not read tau tand Dmax, set them from __init__ input
        with np.load(file) as data:
            self._lambda1 = data["_SU2x2_lambda1"]
            self._lambda2 = data["_SU2x2_lambda2"]
            self._lambda3 = data["_SU2x2_lambda3"]
            self._lambda4 = data["_SU2x2_lambda4"]
            self._lambda5 = data["_SU2x2_lambda5"]
            self._lambda6 = data["_SU2x2_lambda6"]
            self._lambda7 = data["_SU2x2_lambda7"]
            self._lambda8 = data["_SU2x2_lambda8"]
            self._colors_p = data["_SU2x2_colors_p"]
            self._colors_a = data["_SU2x2_colors_a"]
            self._colors1 = data["_SU2x2_colors1"]
            self._colors2 = data["_SU2x2_colors2"]
            self._colors3 = data["_SU2x2_colors3"]
            self._colors4 = data["_SU2x2_colors4"]
            self._colors5 = data["_SU2x2_colors5"]
            self._colors6 = data["_SU2x2_colors6"]
            self._colors7 = data["_SU2x2_colors7"]
            self._colors8 = data["_SU2x2_colors8"]
            self._gammaA = data["_SU2x2_gammaA"]
            self._gammaB = data["_SU2x2_gammaB"]
            self._gammaC = data["_SU2x2_gammaC"]
            self._gammaD = data["_SU2x2_gammaD"]
            self._eigvals_h1 = data["_SU2x2_eigvals_h1"]
            self._eigvecs_h1 = data["_SU2x2_eigvecs_h1"]
            self._eigvals_h2 = data["_SU2x2_eigvals_h2"]
            self._eigvecs_h2 = data["_SU2x2_eigvecs_h2"]
            self.tau = data["_SU2x2_tau"][()]
            self._beta = data["_SU2x2_beta"][()]
            self.Dmax = data["_SU2x2_Dmax"][()]
            if "_SU2x2_cutoff" in data.files:
                self.cutoff = data["_SU2x2_cutoff"][()]
            else:
                self.cutoff = 1e-13  # default value for backward compatibility
            if "_SU2x2_degen_ratio" in data.files:
                self.degen_ratio = data["_SU2x2_degen_ratio"][()]
            else:
                self.degen_ratio = None
        self._D1 = self._lambda1.size
        self._D2 = self._lambda2.size
        self._D3 = self._lambda3.size
        self._D4 = self._lambda4.size
        self._D5 = self._lambda5.size
        self._D6 = self._lambda6.size
        self._D7 = self._lambda7.size
        self._D8 = self._lambda8.size

        if self._d != self._gammaA.shape[0]:
            raise ValueError("Physical dimension differs from save")
        if self._a != self._gammaA.shape[1]:
            raise ValueError("Ancila dimension differs from save")

    def save_to_file(self, file=None):
        data = {}
        data["_SU2x2_lambda1"] = self._lambda1
        data["_SU2x2_lambda2"] = self._lambda2
        data["_SU2x2_lambda3"] = self._lambda3
        data["_SU2x2_lambda4"] = self._lambda4
        data["_SU2x2_lambda5"] = self._lambda5
        data["_SU2x2_lambda6"] = self._lambda6
        data["_SU2x2_lambda7"] = self._lambda7
        data["_SU2x2_lambda8"] = self._lambda8
        data["_SU2x2_colors_p"] = self._colors_p
        data["_SU2x2_colors_a"] = self._colors_a
        data["_SU2x2_colors1"] = self._colors1
        data["_SU2x2_colors2"] = self._colors2
        data["_SU2x2_colors3"] = self._colors3
        data["_SU2x2_colors4"] = self._colors4
        data["_SU2x2_colors5"] = self._colors5
        data["_SU2x2_colors6"] = self._colors6
        data["_SU2x2_colors7"] = self._colors7
        data["_SU2x2_colors8"] = self._colors8
        data["_SU2x2_gammaA"] = self._gammaA
        data["_SU2x2_gammaB"] = self._gammaB
        data["_SU2x2_gammaC"] = self._gammaC
        data["_SU2x2_gammaD"] = self._gammaD
        data["_SU2x2_eigvals_h1"] = self._eigvals_h1
        data["_SU2x2_eigvecs_h1"] = self._eigvecs_h1
        data["_SU2x2_eigvals_h2"] = self._eigvals_h2
        data["_SU2x2_eigvecs_h2"] = self._eigvecs_h2
        data["_SU2x2_tau"] = self._tau
        data["_SU2x2_beta"] = self._beta
        data["_SU2x2_Dmax"] = self.Dmax
        data["_SU2x2_cutoff"] = self.cutoff
        if self.degen_ratio is not None:
            data["_SU2x2_degen_ratio"] = self.degen_ratio
        if file is None:
            return data
        np.savez_compressed(file, **data)
        if self.verbosity > 0:
            print("Simple update data stored in file", file)

    def get_ABCD(self):
        """
        Return optimized tensors A, B, C and D.
        Tensors are obtained by adding relevant sqrt(lambda) to every leg of gammaX
        """
        # actually weights are on by default, so *remove* sqrt(lambda)
        sl1 = 1 / np.sqrt(self._lambda1)
        sl2 = 1 / np.sqrt(self._lambda2)
        sl3 = 1 / np.sqrt(self._lambda3)
        sl4 = 1 / np.sqrt(self._lambda4)
        sl5 = 1 / np.sqrt(self._lambda5)
        sl6 = 1 / np.sqrt(self._lambda6)
        sl7 = 1 / np.sqrt(self._lambda7)
        sl8 = 1 / np.sqrt(self._lambda8)
        A = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaA, sl1, sl2, sl3, sl4)
        B = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaB, sl5, sl4, sl6, sl2)
        C = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaC, sl3, sl7, sl1, sl8)
        D = np.einsum("paurdl,u,r,d,l->paurdl", self._gammaD, sl6, sl8, sl5, sl7)
        A /= np.amax(A)
        B /= np.amax(B)
        C /= np.amax(C)
        D /= np.amax(D)
        return A, B, C, D

    def get_colors_ABCD(self):
        """
        Return colors of optimized tensors A, B, C and D.
        """
        colorsA = (
            self._colors_p,
            self._colors_a,
            self._colors1,
            self._colors2,
            self._colors3,
            self._colors4,
        )
        colorsB = (
            -self._colors_p,
            -self._colors_a,
            -self._colors5,
            -self._colors4,
            -self._colors6,
            -self._colors2,
        )
        colorsC = (
            -self._colors_p,
            -self._colors_a,
            -self._colors3,
            -self._colors7,
            -self._colors1,
            -self._colors8,
        )
        colorsD = (
            self._colors_p,
            self._colors_a,
            self._colors6,
            self._colors8,
            self._colors5,
            self._colors7,
        )
        return colorsA, colorsB, colorsC, colorsD

    def update_first_neighbor(self):
        """
        Update all first neighbor links with first order Trotter-Suzuki
        """
        if self.verbosity > 0:
            print("launch first neighbor update")
        self.update_bond1(self._g1)  # AC up
        self.update_bond2(self._g1)  # AB left
        self.update_bond3(self._g1)  # AC down
        self.update_bond4(self._g1)  # AB right
        self.update_bond5(self._g1)  # BD up
        self.update_bond6(self._g1)  # BD down
        self.update_bond7(self._g1)  # CD left
        self.update_bond8(self._g1)  # CD right

    def update_second_neighbor(self):
        """
        Update all first neighbor links with first order Trotter-Suzuki
        """
        if self.verbosity > 0:
            print("launch second neighbor update")
        # link AD right up
        self.update_bonds25(self._g2_sqrt)  # through B
        self.update_bonds17(self._g2_sqrt)  # through C
        # link AD right down
        self.update_bonds26(self._g2_sqrt)  # through B
        self.update_bonds37(self._g2_sqrt)  # through C
        # link AD left down
        self.update_bonds46(self._g2_sqrt)  # through B
        self.update_bonds38(self._g2_sqrt)  # through C
        # link AD left up
        self.update_bonds45(self._g2_sqrt)  # through B
        self.update_bonds18(self._g2_sqrt)  # through C

        # link BC right up
        self.update_bonds41(self._g2_sqrt)  # through A
        self.update_bonds58(self._g2_sqrt)  # through D
        # link BC right down
        self.update_bonds43(self._g2_sqrt)  # through A
        self.update_bonds68(self._g2_sqrt)  # through D
        # link BC left down
        self.update_bonds23(self._g2_sqrt)  # through A
        self.update_bonds67(self._g2_sqrt)  # through D
        # link BC left up
        self.update_bonds21(self._g2_sqrt)  # through A
        self.update_bonds57(self._g2_sqrt)  # through D

    def update_first_order(self):
        """
        First order Trotter Suzuki step on all bonds.
        """
        if self.verbosity > 0:
            print("launch first order update for all bonds")
        # goes to 2nd order Trotter by reversing update order
        self.update_first_neighbor()
        self.update_second_neighbor()
        self._beta = round(self._beta + self._tau, 10)

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

        self.update_bond1(self._g1)
        for i in range(niter - 1):  # there is 1 step out of the loop
            self._2nd_order_step_no1()
            self.update_bond1(self._g1_squared)
        self._2nd_order_step_no1()
        self.update_bond1(self._g1)
        self._beta = round(self._beta + 4 * niter * self._tau, 10)

    def _2nd_order_step_no1(self):
        """
        Auxiliary function to perform second order Trotter-Suzuki step. Call all update
        but bond1 and bonds57, then bonds57 with g2**2, then all other updates in
        reverse order. Bond1 has to be updated outside of this function.
        """
        self.update_bond2(self._g1)
        self.update_bond3(self._g1)
        self.update_bond4(self._g1)
        self.update_bond5(self._g1)
        self.update_bond6(self._g1)
        self.update_bond7(self._g1)
        self.update_bond8(self._g1)
        self.update_bonds25(self._g2_sqrt)
        self.update_bonds17(self._g2_sqrt)
        self.update_bonds26(self._g2_sqrt)
        self.update_bonds37(self._g2_sqrt)
        self.update_bonds46(self._g2_sqrt)
        self.update_bonds38(self._g2_sqrt)
        self.update_bonds45(self._g2_sqrt)
        self.update_bonds18(self._g2_sqrt)
        self.update_bonds41(self._g2_sqrt)
        self.update_bonds58(self._g2_sqrt)
        self.update_bonds43(self._g2_sqrt)
        self.update_bonds68(self._g2_sqrt)
        self.update_bonds23(self._g2_sqrt)
        self.update_bonds67(self._g2_sqrt)
        self.update_bonds21(self._g2_sqrt)
        self.update_bonds57(self._g2)
        self.update_bonds21(self._g2_sqrt)
        self.update_bonds67(self._g2_sqrt)
        self.update_bonds23(self._g2_sqrt)
        self.update_bonds68(self._g2_sqrt)
        self.update_bonds43(self._g2_sqrt)
        self.update_bonds58(self._g2_sqrt)
        self.update_bonds41(self._g2_sqrt)
        self.update_bonds18(self._g2_sqrt)
        self.update_bonds45(self._g2_sqrt)
        self.update_bonds38(self._g2_sqrt)
        self.update_bonds46(self._g2_sqrt)
        self.update_bonds37(self._g2_sqrt)
        self.update_bonds26(self._g2_sqrt)
        self.update_bonds17(self._g2_sqrt)
        self.update_bonds25(self._g2_sqrt)
        self.update_bond8(self._g1)
        self.update_bond7(self._g1)
        self.update_bond6(self._g1)
        self.update_bond5(self._g1)
        self.update_bond4(self._g1)
        self.update_bond3(self._g1)
        self.update_bond2(self._g1)

    def bond_entanglement_entropy(self):
        """
        Compute the entanglement entropy on every bonds as S = -sum_i p_i log_pi
        """
        s = np.empty(8)
        s[0] = -self._lambda1 @ np.log(self._lambda1)
        s[1] = -self._lambda2 @ np.log(self._lambda2)
        s[2] = -self._lambda3 @ np.log(self._lambda3)
        s[3] = -self._lambda4 @ np.log(self._lambda4)
        s[4] = -self._lambda5 @ np.log(self._lambda5)
        s[5] = -self._lambda6 @ np.log(self._lambda6)
        s[6] = -self._lambda7 @ np.log(self._lambda7)
        s[7] = -self._lambda8 @ np.log(self._lambda8)
        return s

    ###############################################################################
    # first neighbor updates
    ###############################################################################
    # due to pi-roation on B and C sites, colors are opposed on those sites
    # the colors returned by update_neighbors need to be reversed
    # mc{i} = minus color i

    def update_bond1(self, gate):
        """
        Update lambda1 between A and C by applying gate g1 to A upper bond.
        """
        M_A = self._gammaA.transpose(1, 3, 4, 5, 0, 2).reshape(
            self._a * self._D2 * self._D3 * self._D4, self._d * self._D1
        )
        M_C = self._gammaC.transpose(4, 0, 1, 2, 3, 5).reshape(
            self._D1 * self._d, self._a * self._D3 * self._D7 * self._D8
        )

        col_L = combine_colors(
            self._colors_a, self._colors2, self._colors3, self._colors4
        )
        col_R = combine_colors(
            self._colors_a, self._colors3, self._colors7, self._colors8
        )
        # construct matrix theta, renormalize bond dimension and get back tensors
        M_A, self._lambda1, M_C, self._colors1 = update_first_neighbor(
            M_A,
            M_C,
            self._lambda1,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors1,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D1 = self._lambda1.size
        # define new gammaA and gammaC from renormalized M_A and M_C
        self._gammaA = M_A.reshape(
            self._a, self._D2, self._D3, self._D4, self._d, self._D1
        ).transpose(4, 0, 5, 1, 2, 3)
        self._gammaC = M_C.reshape(
            self._D1, self._d, self._a, self._D3, self._D7, self._D8
        ).transpose(1, 2, 3, 4, 0, 5)

        if self.verbosity > 1:
            print("updated bond 1: new lambda1 =", self._lambda1)

    def update_bond2(self, gate):
        """
        Update lambda2 between A and B by applying gate to A right bond.
        """
        M_A = self._gammaA.transpose(1, 2, 4, 5, 0, 3).reshape(
            self._a * self._D1 * self._D3 * self._D4, self._d * self._D2
        )
        M_B = self._gammaB.transpose(5, 0, 1, 2, 3, 4).reshape(
            self._D2 * self._d, self._a * self._D5 * self._D4 * self._D6
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors3, self._colors4
        )
        col_R = combine_colors(
            self._colors_a, self._colors5, self._colors4, self._colors6
        )
        M_A, self._lambda2, M_B, self._colors2 = update_first_neighbor(
            M_A,
            M_B,
            self._lambda2,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors2,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D2 = self._lambda2.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D3, self._D4, self._d, self._D2
        ).transpose(4, 0, 1, 5, 2, 3)
        self._gammaB = M_B.reshape(
            self._D2, self._d, self._a, self._D5, self._D4, self._D6
        ).transpose(1, 2, 3, 4, 5, 0)

        if self.verbosity > 1:
            print("updated bond 2: new lambda2 =", self._lambda2)

    def update_bond3(self, gate):
        """
        Update lambda3 between A and C by applying gate to A down bond.
        """
        M_A = self._gammaA.transpose(1, 2, 3, 5, 0, 4).reshape(
            self._a * self._D1 * self._D2 * self._D4, self._d * self._D3
        )
        M_C = self._gammaC.transpose(2, 0, 1, 3, 4, 5).reshape(
            self._D3 * self._d, self._a * self._D7 * self._D1 * self._D8
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors4
        )
        col_R = combine_colors(
            self._colors_a, self._colors7, self._colors1, self._colors8
        )
        M_A, self._lambda3, M_C, self._colors3 = update_first_neighbor(
            M_A,
            M_C,
            self._lambda3,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors3,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D3 = self._lambda3.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D4, self._d, self._D3
        ).transpose(4, 0, 1, 2, 5, 3)
        self._gammaC = M_C.reshape(
            self._D3, self._d, self._a, self._D7, self._D1, self._D8
        ).transpose(1, 2, 0, 3, 4, 5)

        if self.verbosity > 1:
            print("updated bond 3: new lambda3 =", self._lambda3)

    def update_bond4(self, gate):
        """
        Update lambda4 between A and B by applying gate to A right bond.
        """
        M_A = self._gammaA.transpose(1, 2, 3, 4, 0, 5).reshape(
            self._a * self._D1 * self._D2 * self._D3, self._d * self._D4
        )
        M_B = self._gammaB.transpose(3, 0, 1, 2, 4, 5).reshape(
            self._D4 * self._d, self._a * self._D5 * self._D6 * self._D2
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors3
        )
        col_R = combine_colors(
            self._colors_a, self._colors5, self._colors6, self._colors2
        )
        M_A, self._lambda4, M_B, self._colors4 = update_first_neighbor(
            M_A,
            M_B,
            self._lambda4,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=self._colors4,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D4 = self._lambda4.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D3, self._d, self._D4
        ).transpose(4, 0, 1, 2, 3, 5)
        self._gammaB = M_B.reshape(
            self._D4, self._d, self._a, self._D5, self._D6, self._D2
        ).transpose(1, 2, 3, 0, 4, 5)

        if self.verbosity > 1:
            print("updated bond 4: new lambda4 =", self._lambda4)

    def update_bond5(self, gate):
        """
        Update lambda5 between B and D by applying gate to B upper bond.
        """
        M_B = self._gammaB.transpose(1, 3, 4, 5, 0, 2).reshape(
            self._a * self._D4 * self._D6 * self._D2, self._d * self._D5
        )
        M_D = self._gammaD.transpose(4, 0, 1, 2, 3, 5).reshape(
            self._D5 * self._d, self._a * self._D6 * self._D8 * self._D7
        )

        col_L = -combine_colors(
            self._colors_a, self._colors4, self._colors6, self._colors2
        )
        col_R = -combine_colors(
            self._colors_a, self._colors6, self._colors8, self._colors7
        )
        M_B, self._lambda5, M_D, mc5 = update_first_neighbor(
            M_B,
            M_D,
            self._lambda5,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=-self._colors5,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D5 = self._lambda5.size
        self._colors5 = -mc5  # B-D
        self._gammaB = M_B.reshape(
            self._a, self._D4, self._D6, self._D2, self._d, self._D5
        ).transpose(4, 0, 5, 1, 2, 3)
        self._gammaD = M_D.reshape(
            self._D5, self._d, self._a, self._D6, self._D8, self._D7
        ).transpose(1, 2, 3, 4, 0, 5)

        if self.verbosity > 1:
            print("updated bond 5: new lambda5 =", self._lambda5)

    def update_bond6(self, gate):
        """
        Update lambda6 between B and D by applying gate to B down bond.
        """
        M_B = self._gammaB.transpose(1, 2, 3, 5, 0, 4).reshape(
            self._a * self._D5 * self._D4 * self._D2, self._d * self._D6
        )
        M_D = self._gammaD.transpose(2, 0, 1, 3, 4, 5).reshape(
            self._D6 * self._d, self._a * self._D8 * self._D5 * self._D7
        )

        col_L = -combine_colors(
            self._colors_a, self._colors5, self._colors4, self._colors2
        )
        col_R = -combine_colors(
            self._colors_a, self._colors8, self._colors5, self._colors7
        )
        M_B, self._lambda6, M_D, mc6 = update_first_neighbor(
            M_B,
            M_D,
            self._lambda6,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=-self._colors6,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D6 = self._lambda6.size
        self._colors6 = -mc6  # B-D
        self._gammaB = M_B.reshape(
            self._a, self._D5, self._D4, self._D2, self._d, self._D6
        ).transpose(4, 0, 1, 2, 5, 3)
        self._gammaD = M_D.reshape(
            self._D6, self._d, self._a, self._D8, self._D5, self._D7
        ).transpose(1, 2, 0, 3, 4, 5)

        if self.verbosity > 1:
            print("updated bond 6: new lambda6 =", self._lambda6)

    def update_bond7(self, gate):
        """
        Update lambda7 between C and D by applying gate to C right bond.
        """
        M_C = self._gammaC.transpose(1, 2, 4, 5, 0, 3).reshape(
            self._a * self._D3 * self._D1 * self._D8, self._d * self._D7
        )
        M_D = self._gammaD.transpose(5, 0, 1, 2, 3, 4).reshape(
            self._D7 * self._d, self._a * self._D6 * self._D8 * self._D5
        )

        col_L = -combine_colors(
            self._colors_a, self._colors3, self._colors1, self._colors8
        )
        col_R = -combine_colors(
            self._colors_a, self._colors6, self._colors8, self._colors5
        )
        M_C, self._lambda7, M_D, mc7 = update_first_neighbor(
            M_C,
            M_D,
            self._lambda7,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=-self._colors7,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D7 = self._lambda7.size
        self._colors7 = -mc7  # C-D
        self._gammaC = M_C.reshape(
            self._a, self._D3, self._D1, self._D8, self._d, self._D7
        ).transpose(4, 0, 1, 5, 2, 3)
        self._gammaD = M_D.reshape(
            self._D7, self._d, self._a, self._D6, self._D8, self._D5
        ).transpose(1, 2, 3, 4, 5, 0)

        if self.verbosity > 1:
            print("updated bond 7: new lambda7 =", self._lambda7)

    def update_bond8(self, gate):
        """
        Update lambda8 between C and D by applying gate to C left bond.
        """
        M_C = self._gammaC.transpose(1, 2, 3, 4, 0, 5).reshape(
            self._a * self._D3 * self._D7 * self._D1, self._d * self._D8
        )
        M_D = self._gammaD.transpose(3, 0, 1, 2, 4, 5).reshape(
            self._D8 * self._d, self._a * self._D6 * self._D5 * self._D7
        )

        col_L = -combine_colors(
            self._colors_a, self._colors3, self._colors7, self._colors1
        )
        col_R = -combine_colors(
            self._colors_a, self._colors6, self._colors5, self._colors7
        )
        M_C, self._lambda8, M_D, mc8 = update_first_neighbor(
            M_C,
            M_D,
            self._lambda8,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_R=col_R,
            col_bond=-self._colors8,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D8 = self._lambda8.size
        self._colors8 = -mc8  # C-D
        self._gammaC = M_C.reshape(
            self._a, self._D3, self._D7, self._D1, self._d, self._D8
        ).transpose(4, 0, 1, 2, 3, 5)
        self._gammaD = M_D.reshape(
            self._D8, self._d, self._a, self._D6, self._D5, self._D7
        ).transpose(1, 2, 3, 0, 4, 5)

        if self.verbosity > 1:
            print("updated bond 8: new lambda8 =", self._lambda8)

    ###############################################################################
    # second neighbor updates
    # bonds A-D
    ###############################################################################
    def update_bonds25(self, gate):
        """
        Update lambda2 and lambda6 by applying gate to A upper-right next nearest
        neighbor bond with D through tensor B. Twin of 17.
        """
        M_A = self._gammaA.transpose(1, 2, 4, 5, 0, 3).reshape(
            self._a * self._D1 * self._D3 * self._D4, self._d * self._D2
        )
        M_B = self._gammaB.transpose(5, 2, 0, 1, 3, 4).reshape(
            self._D2 * self._D5, self._d * self._a * self._D4 * self._D6
        )
        M_D = self._gammaD.transpose(4, 0, 1, 2, 3, 5).reshape(
            self._D5 * self._d, self._a * self._D6 * self._D8 * self._D7
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors3, self._colors4
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors4, self._colors6
        )
        col_R = combine_colors(
            self._colors_a, self._colors6, self._colors8, self._colors7
        )
        (
            M_A,
            M_B,
            M_D,
            self._lambda2,
            self._lambda5,
            self._colors2,
            self._colors5,
        ) = update_second_neighbor(
            M_A,
            M_B,
            M_D,
            self._lambda2,
            self._lambda5,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors2,
            col_bR=self._colors5,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D2 = self._lambda2.size
        self._D5 = self._lambda5.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D3, self._D4, self._d, self._D2
        ).transpose(4, 0, 1, 5, 2, 3)
        self._gammaB = M_B.reshape(
            self._D2, self._D5, self._d, self._a, self._D4, self._D6
        ).transpose(2, 3, 1, 4, 5, 0)
        self._gammaD = M_D.reshape(
            self._D5, self._d, self._a, self._D6, self._D8, self._D7
        ).transpose(1, 2, 3, 4, 0, 5)

        if self.verbosity > 1:
            print("updated bonds 2 and 5: new lambda2 =", self._lambda2)
            print("                       new lambda5 =", self._lambda5)

    def update_bonds17(self, gate):
        """
        Update lambda2 and lambda6 by applying gate to A upper-right next nearest
        neighbor bond with D through tensor C. Twin of 25.
        """
        M_A = self._gammaA.transpose(1, 3, 4, 5, 0, 2).reshape(
            self._a * self._D2 * self._D3 * self._D4, self._d * self._D1
        )
        M_C = self._gammaC.transpose(4, 3, 0, 1, 2, 5).reshape(
            self._D1 * self._D7, self._d * self._a * self._D3 * self._D8
        )
        M_D = self._gammaD.transpose(5, 0, 1, 2, 3, 4).reshape(
            self._D7 * self._d, self._a * self._D6 * self._D8 * self._D5
        )

        col_L = combine_colors(
            self._colors_a, self._colors2, self._colors3, self._colors4
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors3, self._colors8
        )
        col_R = combine_colors(
            self._colors_a, self._colors6, self._colors8, self._colors5
        )
        (
            M_A,
            M_C,
            M_D,
            self._lambda1,
            self._lambda7,
            self._colors1,
            self._colors7,
        ) = update_second_neighbor(
            M_A,
            M_C,
            M_D,
            self._lambda1,
            self._lambda7,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors1,
            col_bR=self._colors7,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D1 = self._lambda1.size
        self._D7 = self._lambda7.size
        self._gammaA = M_A.reshape(
            self._a, self._D2, self._D3, self._D4, self._d, self._D1
        ).transpose(4, 0, 5, 1, 2, 3)
        self._gammaC = M_C.reshape(
            self._D1, self._D7, self._d, self._a, self._D3, self._D8
        ).transpose(2, 3, 4, 1, 0, 5)
        self._gammaD = M_D.reshape(
            self._D7, self._d, self._a, self._D6, self._D8, self._D5
        ).transpose(1, 2, 3, 4, 5, 0)

        if self.verbosity > 1:
            print("updated bonds 1 and 7: new lambda1 =", self._lambda1)
            print("                       new lambda7 =", self._lambda7)

    def update_bonds26(self, gate):
        """
        Update lambda2 and lambda6 by applying gate to A down-right next nearest
        neighbor bond with D through tensor B. Twin of 37.
        """
        M_A = self._gammaA.transpose(1, 2, 4, 5, 0, 3).reshape(
            self._a * self._D1 * self._D3 * self._D4, self._d * self._D2
        )
        M_B = self._gammaB.transpose(5, 4, 0, 1, 2, 3).reshape(
            self._D2 * self._D6, self._d * self._a * self._D5 * self._D4
        )
        M_D = self._gammaD.transpose(2, 0, 1, 3, 4, 5).reshape(
            self._D6 * self._d, self._a * self._D8 * self._D5 * self._D7
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors3, self._colors4
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors5, self._colors4
        )
        col_R = combine_colors(
            self._colors_a, self._colors8, self._colors5, self._colors7
        )
        (
            M_A,
            M_B,
            M_D,
            self._lambda2,
            self._lambda6,
            self._colors2,
            self._colors6,
        ) = update_second_neighbor(
            M_A,
            M_B,
            M_D,
            self._lambda2,
            self._lambda6,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors2,
            col_bR=self._colors6,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D2 = self._lambda2.size
        self._D6 = self._lambda6.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D3, self._D4, self._d, self._D2
        ).transpose(4, 0, 1, 5, 2, 3)
        self._gammaB = M_B.reshape(
            self._D2, self._D6, self._d, self._a, self._D5, self._D4
        ).transpose(2, 3, 4, 5, 1, 0)
        self._gammaD = M_D.reshape(
            self._D5, self._d, self._a, self._D6, self._D8, self._D7
        ).transpose(1, 2, 0, 3, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 2 and 6: new lambda2 =", self._lambda2)
            print("                       new lambda6 =", self._lambda6)

    def update_bonds37(self, gate):
        """
        Update lambda2 and lambda6 by applying gate to A down-right next nearest
        neighbor bond with D through tensor C. Twin of 26.
        """
        M_A = self._gammaA.transpose(1, 2, 3, 5, 0, 4).reshape(
            self._a * self._D1 * self._D2 * self._D4, self._d * self._D3
        )
        M_C = self._gammaC.transpose(2, 3, 0, 1, 4, 5).reshape(
            self._D3 * self._D7, self._d * self._a * self._D1 * self._D8
        )
        M_D = self._gammaD.transpose(5, 0, 1, 2, 3, 4).reshape(
            self._D7 * self._d, self._a * self._D6 * self._D8 * self._D5
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors4
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors1, self._colors8
        )
        col_R = combine_colors(
            self._colors_a, self._colors6, self._colors8, self._colors5
        )
        (
            M_A,
            M_C,
            M_D,
            self._lambda3,
            self._lambda7,
            self._colors3,
            self._colors7,
        ) = update_second_neighbor(
            M_A,
            M_C,
            M_D,
            self._lambda3,
            self._lambda7,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors3,
            col_bR=self._colors7,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D3 = self._lambda3.size
        self._D7 = self._lambda7.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D4, self._d, self._D3
        ).transpose(4, 0, 1, 2, 5, 3)
        self._gammaC = M_C.reshape(
            self._D3, self._D7, self._d, self._a, self._D1, self._D8
        ).transpose(2, 3, 0, 1, 4, 5)
        self._gammaD = M_D.reshape(
            self._D7, self._d, self._a, self._D6, self._D8, self._D5
        ).transpose(1, 2, 3, 4, 5, 0)

        if self.verbosity > 1:
            print("updated bonds 3 and 7: new lambda3 =", self._lambda3)
            print("                       new lambda7 =", self._lambda7)

    def update_bonds46(self, gate):
        """
        Update lambda4 and lambda6 by applying gate to A down-left next nearest
        neighbor bond with D through tensor B. Twin of 38.
        """
        M_A = self._gammaA.transpose(1, 2, 3, 4, 0, 5).reshape(
            self._a * self._D1 * self._D2 * self._D3, self._d * self._D4
        )
        M_B = self._gammaB.transpose(3, 4, 0, 1, 2, 5).reshape(
            self._D4 * self._D6, self._d * self._a * self._D5 * self._D2
        )
        M_D = self._gammaD.transpose(2, 0, 1, 3, 4, 5).reshape(
            self._D6 * self._d, self._a * self._D8 * self._D5 * self._D7
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors3
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors5, self._colors2
        )
        col_R = combine_colors(
            self._colors_a, self._colors8, self._colors5, self._colors7
        )
        (
            M_A,
            M_B,
            M_D,
            self._lambda4,
            self._lambda6,
            self._colors4,
            self._colors6,
        ) = update_second_neighbor(
            M_A,
            M_B,
            M_D,
            self._lambda4,
            self._lambda6,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors4,
            col_bR=self._colors6,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D4 = self._lambda4.size
        self._D6 = self._lambda6.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D3, self._d, self._D4
        ).transpose(4, 0, 1, 2, 3, 5)
        self._gammaB = M_B.reshape(
            self._D4, self._D6, self._d, self._a, self._D5, self._D2
        ).transpose(2, 3, 4, 0, 1, 5)
        self._gammaD = M_D.reshape(
            self._D6, self._d, self._a, self._D8, self._D5, self._D7
        ).transpose(1, 2, 0, 3, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 4 and 6: new lambda4 =", self._lambda4)
            print("                       new lambda6 =", self._lambda6)

    def update_bonds38(self, gate):
        """
        Update lambda2 and lambda6 by applying gate to A down-left next nearest
        neighbor bond with D through tensor C. Twin of 46.
        """
        M_A = self._gammaA.transpose(1, 2, 3, 5, 0, 4).reshape(
            self._a * self._D1 * self._D2 * self._D4, self._d * self._D3
        )
        M_C = self._gammaC.transpose(2, 5, 0, 1, 3, 4).reshape(
            self._D3 * self._D8, self._d * self._a * self._D7 * self._D1
        )
        M_D = self._gammaD.transpose(3, 0, 1, 2, 4, 5).reshape(
            self._D8 * self._d, self._a * self._D6 * self._D5 * self._D7
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors4
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors7, self._colors1
        )
        col_R = combine_colors(
            self._colors_a, self._colors6, self._colors5, self._colors7
        )
        (
            M_A,
            M_C,
            M_D,
            self._lambda3,
            self._lambda8,
            self._colors3,
            self._colors8,
        ) = update_second_neighbor(
            M_A,
            M_C,
            M_D,
            self._lambda3,
            self._lambda8,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors3,
            col_bR=self._colors8,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D3 = self._lambda3.size
        self._D8 = self._lambda8.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D4, self._d, self._D3
        ).transpose(4, 0, 1, 2, 5, 3)
        self._gammaC = M_C.reshape(
            self._D3, self._D8, self._d, self._a, self._D7, self._D1
        ).transpose(2, 3, 0, 4, 5, 1)
        self._gammaD = M_D.reshape(
            self._D8, self._d, self._a, self._D6, self._D5, self._D7
        ).transpose(1, 2, 3, 0, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 3 and 8: new lambda3 =", self._lambda3)
            print("                       new lambda8 =", self._lambda8)

    def update_bonds45(self, gate):
        """
        Update lambda4 and lambda5 by applying gate to A upper-left next nearest
        neighbor bond with D through tensor B. Twin of 18.
        """
        M_A = self._gammaA.transpose(1, 2, 3, 4, 0, 5).reshape(
            self._a * self._D1 * self._D2 * self._D3, self._d * self._D4
        )
        M_B = self._gammaB.transpose(3, 2, 0, 1, 4, 5).reshape(
            self._D4 * self._D5, self._d * self._a * self._D6 * self._D2
        )
        M_D = self._gammaD.transpose(4, 0, 1, 2, 3, 5).reshape(
            self._D5 * self._d, self._a * self._D6 * self._D8 * self._D7
        )

        col_L = combine_colors(
            self._colors_a, self._colors1, self._colors2, self._colors3
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors6, self._colors2
        )
        col_R = combine_colors(
            self._colors_a, self._colors6, self._colors8, self._colors7
        )
        (
            M_A,
            M_B,
            M_D,
            self._lambda4,
            self._lambda5,
            self._colors4,
            self._colors5,
        ) = update_second_neighbor(
            M_A,
            M_B,
            M_D,
            self._lambda4,
            self._lambda5,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors4,
            col_bR=self._colors5,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D4 = self._lambda4.size
        self._D5 = self._lambda5.size
        self._gammaA = M_A.reshape(
            self._a, self._D1, self._D2, self._D3, self._d, self._D4
        ).transpose(4, 0, 1, 2, 3, 5)
        self._gammaB = M_B.reshape(
            self._D4, self._D5, self._d, self._a, self._D6, self._D2
        ).transpose(2, 3, 1, 0, 4, 5)
        self._gammaD = M_D.reshape(
            self._D5, self._d, self._a, self._D6, self._D8, self._D7
        ).transpose(1, 2, 3, 4, 0, 5)

        if self.verbosity > 1:
            print("updated bonds 4 and 5: new lambda4 =", self._lambda4)
            print("                       new lambda5 =", self._lambda5)

    def update_bonds18(self, gate):
        """
        Update lambda1 and lambda8 by applying gate to A upper-left next nearest
        neighbor bond with D through tensor C. Twin of 45.
        """
        M_A = self._gammaA.transpose(1, 3, 4, 5, 0, 2).reshape(
            self._a * self._D2 * self._D3 * self._D4, self._d * self._D1
        )
        M_C = self._gammaC.transpose(4, 5, 0, 1, 2, 3).reshape(
            self._D1 * self._D8, self._d * self._a * self._D3 * self._D7
        )
        M_D = self._gammaD.transpose(3, 0, 1, 2, 4, 5).reshape(
            self._D8 * self._d, self._a * self._D6 * self._D5 * self._D7
        )

        col_L = combine_colors(
            self._colors_a, self._colors2, self._colors3, self._colors4
        )
        col_mid = combine_colors(
            self._colors_p, self._colors_a, self._colors3, self._colors7
        )
        col_R = combine_colors(
            self._colors_a, self._colors6, self._colors5, self._colors7
        )
        (
            M_A,
            M_C,
            M_D,
            self._lambda1,
            self._lambda8,
            self._colors1,
            self._colors8,
        ) = update_second_neighbor(
            M_A,
            M_C,
            M_D,
            self._lambda1,
            self._lambda8,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=self._colors1,
            col_bR=self._colors8,
            col_d=self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D1 = self._lambda1.size
        self._D8 = self._lambda8.size
        self._gammaA = M_A.reshape(
            self._a, self._D2, self._D3, self._D4, self._d, self._D1
        ).transpose(4, 0, 5, 1, 2, 3)
        self._gammaC = M_C.reshape(
            self._D1, self._D8, self._d, self._a, self._D3, self._D7
        ).transpose(2, 3, 4, 5, 0, 1)
        self._gammaD = M_D.reshape(
            self._D8, self._d, self._a, self._D6, self._D5, self._D7
        ).transpose(1, 2, 3, 0, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 1 and 8: new lambda1 =", self._lambda1)
            print("                       new lambda8 =", self._lambda8)

    ###############################################################################
    # second neighbor updates
    # bonds B-C
    ###############################################################################
    def update_bonds41(self, gate):
        """
        Update lambda4 and lambda1 by applying gate to B upper-right next nearest
        neighbor bond with C through tensor A. Twin of 58.
        """
        M_B = self._gammaB.transpose(1, 2, 4, 5, 0, 3).reshape(
            self._a * self._D5 * self._D6 * self._D2, self._d * self._D4
        )
        M_A = self._gammaA.transpose(5, 2, 0, 1, 3, 4).reshape(
            self._D4 * self._D1, self._d * self._a * self._D2 * self._D3
        )
        M_C = self._gammaC.transpose(4, 0, 1, 2, 3, 5).reshape(
            self._D1 * self._d, self._a * self._D3 * self._D7 * self._D8
        )

        col_L = -combine_colors(
            self._colors_a, self._colors5, self._colors6, self._colors2
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors2, self._colors3
        )
        col_R = -combine_colors(
            self._colors_a, self._colors3, self._colors7, self._colors8
        )
        M_B, M_A, M_C, self._lambda4, self._lambda1, mc4, mc1 = update_second_neighbor(
            M_B,
            M_A,
            M_C,
            self._lambda4,
            self._lambda1,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors4,
            col_bR=-self._colors1,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D4 = self._lambda4.size
        self._D1 = self._lambda1.size
        self._colors4 = -mc4
        self._colors1 = -mc1
        self._gammaB = M_B.reshape(
            self._a, self._D5, self._D6, self._D2, self._d, self._D4
        ).transpose(4, 0, 1, 5, 2, 3)
        self._gammaA = M_A.reshape(
            self._D4, self._D1, self._d, self._a, self._D2, self._D3
        ).transpose(2, 3, 1, 4, 5, 0)
        self._gammaC = M_C.reshape(
            self._D1, self._d, self._a, self._D3, self._D7, self._D8
        ).transpose(1, 2, 3, 4, 0, 5)

        if self.verbosity > 1:
            print("updated bonds 4 and 1: new lambda4 =", self._lambda4)
            print("                       new lambda1 =", self._lambda1)

    def update_bonds58(self, gate):
        """
        Update lambda2 and lambda6 by applying gate to B upper-right next nearest
        neighbor bond with C through tensor D. Twin of 41.
        """
        M_B = self._gammaB.transpose(1, 3, 4, 5, 0, 2).reshape(
            self._a * self._D4 * self._D6 * self._D2, self._d * self._D5
        )
        M_D = self._gammaD.transpose(4, 3, 0, 1, 2, 5).reshape(
            self._D5 * self._D8, self._d * self._a * self._D6 * self._D7
        )
        M_C = self._gammaC.transpose(5, 0, 1, 2, 3, 4).reshape(
            self._D8 * self._d, self._a * self._D3 * self._D7 * self._D1
        )

        col_L = -combine_colors(
            self._colors_a, self._colors4, self._colors6, self._colors2
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors6, self._colors7
        )
        col_R = -combine_colors(
            self._colors_a, self._colors3, self._colors7, self._colors1
        )
        M_B, M_D, M_C, self._lambda5, self._lambda8, mc5, mc8 = update_second_neighbor(
            M_B,
            M_D,
            M_C,
            self._lambda5,
            self._lambda8,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors5,
            col_bR=-self._colors8,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D5 = self._lambda5.size
        self._D8 = self._lambda8.size
        self._colors5 = -mc5
        self._colors8 = -mc8

        self._gammaB = M_B.reshape(
            self._a, self._D4, self._D6, self._D2, self._d, self._D5
        ).transpose(4, 0, 5, 1, 2, 3)
        self._gammaD = M_D.reshape(
            self._D5, self._D8, self._d, self._a, self._D6, self._D7
        ).transpose(2, 3, 4, 1, 0, 5)
        self._gammaC = M_C.reshape(
            self._D8, self._d, self._a, self._D3, self._D7, self._D1
        ).transpose(1, 2, 3, 4, 5, 0)

        if self.verbosity > 1:
            print("updated bonds 5 and 8: new lambda5 =", self._lambda5)
            print("                       new lambda8 =", self._lambda8)

    def update_bonds43(self, gate):
        """
        Update lambda4 and lambda3 by applying gate to B down-right next nearest
        neighbor bond with C through tensor A. Twin of 68.
        """
        M_B = self._gammaB.transpose(1, 2, 4, 5, 0, 3).reshape(
            self._a * self._D5 * self._D6 * self._D2, self._d * self._D4
        )
        M_A = self._gammaA.transpose(5, 4, 0, 1, 2, 3).reshape(
            self._D4 * self._D3, self._d * self._a * self._D1 * self._D2
        )
        M_C = self._gammaC.transpose(2, 0, 1, 3, 4, 5).reshape(
            self._D3 * self._d, self._a * self._D7 * self._D1 * self._D8
        )

        col_L = -combine_colors(
            self._colors_a, self._colors5, self._colors6, self._colors2
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors1, self._colors2
        )
        col_R = -combine_colors(
            self._colors_a, self._colors7, self._colors1, self._colors8
        )
        M_B, M_A, M_C, self._lambda4, self._lambda3, mc4, mc3 = update_second_neighbor(
            M_B,
            M_A,
            M_C,
            self._lambda4,
            self._lambda3,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors4,
            col_bR=-self._colors3,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )

        self._D4 = self._lambda4.size
        self._D3 = self._lambda3.size
        self._colors4 = -mc4
        self._colors3 = -mc3
        self._gammaB = M_B.reshape(
            self._a, self._D5, self._D6, self._D2, self._d, self._D4
        ).transpose(4, 0, 1, 5, 2, 3)
        self._gammaA = M_A.reshape(
            self._D4, self._D3, self._d, self._a, self._D1, self._D2
        ).transpose(2, 3, 4, 5, 1, 0)
        self._gammaC = M_C.reshape(
            self._D3, self._d, self._a, self._D7, self._D1, self._D8
        ).transpose(1, 2, 0, 3, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 4 and 3: new lambda4 =", self._lambda4)
            print("                       new lambda3 =", self._lambda3)

    def update_bonds68(self, gate):
        """
        Update lambda2 and lambda6 by applying gate to B down-right next nearest
        neighbor bond with C through tensor D. Twin of 43.
        """
        M_B = self._gammaB.transpose(1, 2, 3, 5, 0, 4).reshape(
            self._a * self._D5 * self._D4 * self._D2, self._d * self._D6
        )
        M_D = self._gammaD.transpose(2, 3, 0, 1, 4, 5).reshape(
            self._D6 * self._D8, self._d * self._a * self._D5 * self._D7
        )
        M_C = self._gammaC.transpose(5, 0, 1, 2, 3, 4).reshape(
            self._D8 * self._d, self._a * self._D3 * self._D7 * self._D1
        )

        col_L = -combine_colors(
            self._colors_a, self._colors5, self._colors4, self._colors2
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors5, self._colors7
        )
        col_R = -combine_colors(
            self._colors_a, self._colors3, self._colors7, self._colors1
        )
        M_B, M_D, M_C, self._lambda6, self._lambda8, mc6, mc8 = update_second_neighbor(
            M_B,
            M_D,
            M_C,
            self._lambda6,
            self._lambda8,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors6,
            col_bR=-self._colors8,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )
        self._D6 = self._lambda6.size
        self._D8 = self._lambda8.size
        self._colors6 = -mc6
        self._colors8 = -mc8

        self._gammaB = M_B.reshape(
            self._a, self._D5, self._D4, self._D2, self._d, self._D6
        ).transpose(4, 0, 1, 2, 5, 3)
        self._gammaD = M_D.reshape(
            self._D6, self._D8, self._d, self._a, self._D5, self._D7
        ).transpose(2, 3, 0, 1, 4, 5)
        self._gammaC = M_C.reshape(
            self._D8, self._d, self._a, self._D3, self._D7, self._D1
        ).transpose(1, 2, 3, 4, 5, 0)

        if self.verbosity > 1:
            print("updated bonds 6 and 8: new lambda6 =", self._lambda6)
            print("                       new lambda8 =", self._lambda8)

    def update_bonds23(self, gate):
        """
        Update lambda2 and lambda3 by applying gate to B down-left next nearest
        neighbor bond with C through tensor A. Twin of 67.
        """
        M_B = self._gammaB.transpose(1, 2, 3, 4, 0, 5).reshape(
            self._a * self._D5 * self._D4 * self._D6, self._d * self._D2
        )
        M_A = self._gammaA.transpose(3, 4, 0, 1, 2, 5).reshape(
            self._D2 * self._D3, self._d * self._a * self._D1 * self._D4
        )
        M_C = self._gammaC.transpose(2, 0, 1, 3, 4, 5).reshape(
            self._D3 * self._d, self._a * self._D7 * self._D1 * self._D8
        )

        col_L = -combine_colors(
            self._colors_a, self._colors5, self._colors4, self._colors6
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors1, self._colors4
        )
        col_R = -combine_colors(
            self._colors_a, self._colors7, self._colors1, self._colors8
        )
        M_B, M_A, M_C, self._lambda2, self._lambda3, mc2, mc3 = update_second_neighbor(
            M_B,
            M_A,
            M_C,
            self._lambda2,
            self._lambda3,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors2,
            col_bR=-self._colors3,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )
        self._D2 = self._lambda2.size
        self._D3 = self._lambda3.size
        self._colors2 = -mc2
        self._colors3 = -mc3

        self._gammaB = M_B.reshape(
            self._a, self._D5, self._D4, self._D6, self._d, self._D2
        ).transpose(4, 0, 1, 2, 3, 5)
        self._gammaA = M_A.reshape(
            self._D2, self._D3, self._d, self._a, self._D1, self._D4
        ).transpose(2, 3, 4, 0, 1, 5)
        self._gammaC = M_C.reshape(
            self._D3, self._d, self._a, self._D7, self._D1, self._D8
        ).transpose(1, 2, 0, 3, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 2 and 3: new lambda2 =", self._lambda2)
            print("                       new lambda3 =", self._lambda3)

    def update_bonds67(self, gate):
        """
        Update lambda6 and lambda7 by applying gate to B down-left next nearest
        neighbor bond with C through tensor D. Twin of 23.
        """
        M_B = self._gammaB.transpose(1, 2, 3, 5, 0, 4).reshape(
            self._a * self._D5 * self._D4 * self._D2, self._d * self._D6
        )
        M_D = self._gammaD.transpose(2, 5, 0, 1, 3, 4).reshape(
            self._D6 * self._D7, self._d * self._a * self._D8 * self._D5
        )
        M_C = self._gammaC.transpose(3, 0, 1, 2, 4, 5).reshape(
            self._D7 * self._d, self._a * self._D3 * self._D1 * self._D8
        )

        col_L = -combine_colors(
            self._colors_a, self._colors5, self._colors4, self._colors2
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors8, self._colors5
        )
        col_R = -combine_colors(
            self._colors_a, self._colors3, self._colors1, self._colors8
        )
        M_B, M_D, M_C, self._lambda6, self._lambda7, mc6, mc7 = update_second_neighbor(
            M_B,
            M_D,
            M_C,
            self._lambda6,
            self._lambda7,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors6,
            col_bR=-self._colors7,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )
        self._D6 = self._lambda6.size
        self._D7 = self._lambda7.size
        self._colors6 = -mc6
        self._colors7 = -mc7

        self._gammaB = M_B.reshape(
            self._a, self._D5, self._D4, self._D2, self._d, self._D6
        ).transpose(4, 0, 1, 2, 5, 3)
        self._gammaD = M_D.reshape(
            self._D6, self._D7, self._d, self._a, self._D8, self._D5
        ).transpose(2, 3, 0, 4, 5, 1)
        self._gammaC = M_C.reshape(
            self._D7, self._d, self._a, self._D3, self._D1, self._D8
        ).transpose(1, 2, 3, 0, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 6 and 7: new lambda6 =", self._lambda6)
            print("                       new lambda7 =", self._lambda7)

    def update_bonds21(self, gate):
        """
        Update lambda2 and lambda1 by applying gate to B upper-left next nearest
        neighbor bond with C through tensor A. Twin of 57.
        """
        M_B = self._gammaB.transpose(1, 2, 3, 4, 0, 5).reshape(
            self._a * self._D5 * self._D4 * self._D6, self._d * self._D2
        )
        M_A = self._gammaA.transpose(3, 2, 0, 1, 4, 5).reshape(
            self._D2 * self._D1, self._d * self._a * self._D3 * self._D4
        )
        M_C = self._gammaC.transpose(4, 0, 1, 2, 3, 5).reshape(
            self._D1 * self._d, self._a * self._D3 * self._D7 * self._D8
        )

        col_L = -combine_colors(
            self._colors_a, self._colors5, self._colors4, self._colors6
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors3, self._colors4
        )
        col_R = -combine_colors(
            self._colors_a, self._colors3, self._colors7, self._colors8
        )
        M_B, M_A, M_C, self._lambda2, self._lambda1, mc2, mc1 = update_second_neighbor(
            M_B,
            M_A,
            M_C,
            self._lambda2,
            self._lambda1,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors2,
            col_bR=-self._colors1,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )
        self._D2 = self._lambda2.size
        self._D1 = self._lambda1.size
        self._colors2 = -mc2
        self._colors1 = -mc1

        self._gammaB = M_B.reshape(
            self._a, self._D5, self._D4, self._D6, self._d, self._D2
        ).transpose(4, 0, 1, 2, 3, 5)
        self._gammaA = M_A.reshape(
            self._D2, self._D1, self._d, self._a, self._D3, self._D4
        ).transpose(2, 3, 1, 0, 4, 5)
        self._gammaC = M_C.reshape(
            self._D1, self._d, self._a, self._D3, self._D7, self._D8
        ).transpose(1, 2, 3, 4, 0, 5)

        if self.verbosity > 1:
            print("updated bonds 2 and 1: new lambda2 =", self._lambda2)
            print("                       new lambda1 =", self._lambda1)

    def update_bonds57(self, gate):
        """
        Update lambda6 and lambda7 by applying gate to B down-left next nearest
        neighbor bond with C through tensor D. Twin of 21.
        """
        M_B = self._gammaB.transpose(1, 3, 4, 5, 0, 2).reshape(
            self._a * self._D4 * self._D6 * self._D2, self._d * self._D5
        )
        M_D = self._gammaD.transpose(4, 5, 0, 1, 2, 3).reshape(
            self._D5 * self._D7, self._d * self._a * self._D6 * self._D8
        )
        M_C = self._gammaC.transpose(3, 0, 1, 2, 4, 5).reshape(
            self._D7 * self._d, self._a * self._D3 * self._D1 * self._D8
        )

        col_L = -combine_colors(
            self._colors_a, self._colors4, self._colors6, self._colors2
        )
        col_mid = -combine_colors(
            self._colors_p, self._colors_a, self._colors6, self._colors8
        )
        col_R = -combine_colors(
            self._colors_a, self._colors3, self._colors1, self._colors8
        )
        M_B, M_D, M_C, self._lambda5, self._lambda7, mc5, mc7 = update_second_neighbor(
            M_B,
            M_D,
            M_C,
            self._lambda5,
            self._lambda7,
            gate,
            self._d,
            self.Dmax,
            col_L=col_L,
            col_mid=col_mid,
            col_R=col_R,
            col_bL=-self._colors5,
            col_bR=-self._colors7,
            col_d=-self._colors_p,
            cutoff=self.cutoff,
            degen_ratio=self.degen_ratio,
        )
        self._D5 = self._lambda5.size
        self._D7 = self._lambda7.size
        self._colors5 = -mc5
        self._colors7 = -mc7

        self._gammaB = M_B.reshape(
            self._a, self._D4, self._D6, self._D2, self._d, self._D5
        ).transpose(4, 0, 5, 1, 2, 3)
        self._gammaD = M_D.reshape(
            self._D5, self._D7, self._d, self._a, self._D6, self._D8
        ).transpose(2, 3, 4, 5, 0, 1)
        self._gammaC = M_C.reshape(
            self._D7, self._d, self._a, self._D3, self._D1, self._D8
        ).transpose(1, 2, 3, 0, 4, 5)

        if self.verbosity > 1:
            print("updated bonds 5 and 7: new lambda5 =", self._lambda5)
            print("                       new lambda7 =", self._lambda7)
