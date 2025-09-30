class FullUpdateAB:

    def __init__(self, ctm, h, beta, tau, logZ, *, verbosity=0):
        self._ctm = ctm
        self._h = h
        self._beta = beta
        self.tau = tau
        self._logZ = logZ
        self.verbosity = verbosity
        self._n_updates = 4
        self._is_second_order = False

    @property
    def Dmax(self):
        return self._ctm.Dmax

    @property
    def beta(self):
        return self._beta

    @property
    def n_bonds(self):
        return 4

    @property
    def n_tensors(self):
        return 2

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self.verbosity > 0:
            print(f"set tau to {tau}")
        self._tau = tau
        self._gate = (-tau * self._h).expm()
        self._gate2 = (-2 * tau * self._h).expm()
        # rho is quadratic in psi
        self._dbeta = 2 * (1 + self._is_second_order) * tau

    @property
    def logZ(self):
        return self._logZ / self._n_tensors

    def evolve(self, beta_evolve):
        """
        Evolve in imaginary time using second order Trotter-Suzuki up to beta.
        Convention: temperature value is the bilayer tensor one, twice the monolayer
        one.
        """
        if self.verbosity > 0:
            print(
                f"Evolve in imaginary time for beta from {self._beta:.6g} to "
                f"{self._beta + beta_evolve:.6g}...",
                flush=True,
            )
        if beta_evolve < -1e-12:
            raise ValueError("Cannot evolve for negative imaginary time")
        if beta_evolve < 0.9 * self._dbeta:  # care for float rounding
            return  # else evolve for 1 step out of niter loop
        niter = round(beta_evolve / self._dbeta)  # 2nd order: evolve 2*tau by step

        # 2nd order update: update 0 is special as it may be squared or not
        # depending whether it is at the very beginning / end of update sequence
        self._initialize_update()
        for _ in range(niter - 1):  # there is 1 step out of the loop
            for j in range(self._n_updates):
                self._elementary_update(j)
        self._finalize_update()
        self._beta += niter * self._dbeta
        return

    def _initialize_update(self):
        # For a given tensor, its structure is defined by tensor_bond_indices between
        # updates. This structure needs to change at each update the tensor is involved
        # in. To avoid swappping back and forth for each elementary update, the tensor
        # structure constraint is relaxed inside evolve and changes within updates, as
        # defined by lperm, rperm and mperm. One cycle of updates will leave tensors
        # in a given final state structure, one needs to start from this state to be
        # able to loop over it.
        for i in range(self._n_tensors):
            self._tensors[i] = self._tensors[i].permute(*self._initial_swap[i])

        # run last elementary update with time step tau instead of 2*tau
        self._elementary_update(self._n_updates)

        # run one cycle of update but the first one
        for j in range(1, self._n_updates):
            self._elementary_update(j)

    def _finalize_update(self):
        if self._is_second_order:
            # run the very last/first update with time step tau
            self._elementary_update(self._n_updates)

        # come back to default form, as defined in tensor_bond_indices
        for i in range(self._n_tensors):
            self._tensors[i] = self._tensors[i].permute(*self._final_swap[i])

    def _elementary_update(self, j):
        cst_left, updated_left, updated_right, cst_right = self.apply_gate(j)
        eff_env = self.compute_env(j, cst_left, cst_right)
        renormalized_left, renormalized_right = self.optimize_tensors(
            eff_env, updated_left, updated_right
        )
        new_left = cst_left @ renormalized_left
        new_right = renormalized_right @ cst_right
        self._set_site_tensors(j, new_left, new_right)

    def apply_gate_up(self):
        left = self._ctm.get_A(1, 1).permute((1, 2, 4, 5), (0, 3))
        right = self._ctm.get_A(2, 0).permute((0, 5), (1, 2, 3, 4))
        return self.apply_gate(left, right)

    def apply_gate(self, left, right):
        cst_left, eff_left = left.qr()
        eff_right, cst_right = right.rq()

        # change tensor structure to contract mid
        permuted_left = eff_left.permute((0, 1), (2,))  # auxL,p=effL-m
        permuted_right = eff_right.permute((2,), (0, 1))  # m-effR=auxR,p

        # construct matrix theta and apply gate
        theta = permuted_left @ permuted_right  # auxL,pL=theta=auxR,pR
        theta = theta.permute((0, 2), (1, 3))  # auxL, auxR = theta = pL, pR
        theta = theta @ self._gate

        theta = theta.permute((0, 2), (1, 3))  # auxL, pL = theta = auxR, pR
        u, new_weights, v = theta.svd()
        sqw = new_weights**0.5
        new_left = u * sqw
        new_right = sqw * v
        return cst_left, new_left, new_right, cst_right

    def compute_right_env(self, cst_left, cst_right):
        x, y = 0, 0
        env0 = contract_env(
            self.ctm.get_C1(x, y),
            self.ctm.get_T1(x + 1, y),
            self.ctm.get_T1(x + 2, y),
            self.ctm.get_C2(x + 3, y),
            self.ctm.get_T4(x, y + 1),
            cst_left,
            cst_right,
            self.ctm.get_T2(x + 3, y + 1),
            self.ctm.get_C4(x, y + 2),
            self.ctm.get_T3(x + 1, y + 2),
            self.ctm.get_T3(x + 2, y + 2),
            self.ctm.get_C3(x + 3, y + 2),
        )
        env = fix_gauge(env0, rtol=self.rtol)




def fix_gauge(env0, *, rtol=1e-16, atol=0.0):
    z = env.cholesky()
    ql, rmat = z.qr()
    lmat, qr = z.rq()
    ztilde = rmat.dagger() @ z @ lmat
    return env


def contract_env(C1, T1l, T1r, C2, T4, cst_left, cst_right, T2, C4, T3l, T3r, C3):
    left_env = T1l.permute((1, 2, 0), (3,)) @ C1
    left_env = left_env @ (T4.permute((0, 1, 2), (3,)) @ C4).permute((0,), (1, 2, 3))

    # -----2        ------4
    # |  ||         |   ||
    # |  01    -->  |   02
    # |=3,4         |=1,3
    # 5             5
    left_env = left_env.permute((0, 3), (1, 4, 2, 5))

    #  C1----T1-4         C1----T1-5
    #  |     ||           |     ||
    #  |     02           |     |3
    #  |     3            |     |
    #  |     |      -->   |     |
    #  T4-14-left-4       T4----left-2
    #  | \3  |\           | \4  |\
    #  5     2 0          6     1 0
    left_env = cst_left.permute((0, 2, 4), (1, 3)) @ left_env
    left_env = left_env.permute((0, 3, 4), (1, 2, 5, 6))

    #  C1----T1-5
    #  |     ||
    #  |     |1
    #  |     |              1
    #  |     |              |
    #  T4----left-3      3-left-4*
    #  | \2  |\             |\
    #  6     4 0            2 0
    left_env = cst_left.permute((0, 1, 3), (2, 4)).dagger() @ left_env

    # -----4         -----2
    # |  ||          |  ||
    # |  ||          |  ||
    # |=====3,1  --> |=====0,1
    # |  ||          |  ||
    # 5  02          5  34
    left_env = left_env.permute((3, 1, 4), (2, 0, 5))
    left_env = left_env @ T3l.permute((0, 1, 3), (2,))
    # ------2
    # |  ||
    # |=====0,1
    # |  ||
    # ------3

    right_env = (T2.permute((0, 2, 3), (1,)) @ C3).permute((0,), (1, 2, 3))
    up = C2 @ T1r.permute((0,), (1, 2, 3))
    right_env = up.transpose() @ right_env
    right_env = right_env.permute((0, 3), (1, 4, 2, 5))
    right_env = cst_right.permute((1, 0, 4), (2, 3)) @ right_env
    right_env = right_env.permute((0, 3, 4), (1, 2, 5, 6))
    right_env = cst_right.permute((1, 2, 3), (0, 4)).dagger() @ right_env

    #   4------
    #      || |
    #      || |
    #  2,0====|
    #      || |
    #      31 5
    right_env = right_env.permute((4, 0, 2), (3, 1, 5)) @ T3r.permute((0, 1, 2), (3,))

    #   0------
    #      || |
    #  1,2====|
    #      || |
    #   3------
    #
    env = left_env.permute((0, 1), (2, 3)) @ right_env.permute((0, 3), (1, 2))
    return env.permute((0, 2), (1, 3))
