import numpy as np

from simple_update.simple_update import SimpleUpdate


class SimpleUpdate2x2(SimpleUpdate):
    """
    Simple update algorithm on plaquette AB//CD, with
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
    _n_hamiltonians = 2
    _n_tensors = 4

    # permutations used in evolve (generated)
    _d2d = ((0, 1, 2, 3), (4, 5))
    _d2l = ((0, 1, 2, 5), (4, 3))
    _d2r = ((0, 1, 5, 3), (4, 2))
    _d2u = ((0, 2, 5, 3), (4, 1))
    _d2ur = ((4, 0, 5, 3), (1, 2))
    _dl2l = ((1, 2, 3, 4), (0, 5))
    _dl2lu = ((0, 1, 3, 4), (5, 2))
    _dl2rd = ((0, 1, 2, 5), (3, 4))
    _dl2u = ((1, 3, 4, 5), (0, 2))
    _l2d = ((0, 1, 2, 5), (4, 3))
    _l2dl = ((4, 0, 1, 2), (3, 5))
    _l2l = ((0, 1, 2, 3), (4, 5))
    _l2r = ((0, 1, 3, 5), (4, 2))
    _l2u = ((0, 2, 3, 5), (4, 1))
    _l2ur = ((4, 0, 3, 5), (1, 2))
    _lu2dl = ((0, 1, 5, 2), (3, 4))
    _lu2r = ((1, 5, 3, 4), (0, 2))
    _lu2ur = ((0, 1, 3, 4), (5, 2))
    _r2d = ((0, 1, 5, 3), (4, 2))
    _r2l = ((0, 1, 5, 2), (4, 3))
    _r2lu = ((4, 0, 5, 2), (3, 1))
    _r2r = ((0, 1, 2, 3), (4, 5))
    _r2u = ((0, 5, 2, 3), (4, 1))
    _rd2dl = ((0, 1, 2, 4), (5, 3))
    _rd2rd = ((0, 1, 2, 3), (4, 5))
    _rd2u = ((1, 4, 5, 3), (0, 2))
    _rd2ur = ((0, 1, 5, 3), (2, 4))
    _u2d = ((0, 5, 1, 3), (4, 2))
    _u2dl = ((4, 0, 5, 1), (2, 3))
    _u2l = ((0, 5, 1, 2), (4, 3))
    _u2r = ((0, 5, 2, 3), (4, 1))
    _u2rd = ((4, 0, 5, 3), (1, 2))
    _u2u = ((0, 1, 2, 3), (4, 5))
    _ur2d = ((1, 4, 5, 3), (0, 2))
    _ur2l = ((1, 4, 5, 2), (0, 3))
    _ur2lu = ((0, 1, 5, 2), (3, 4))
    _ur2rd = ((0, 1, 4, 3), (5, 2))

    @classmethod
    def from_infinite_temperature(
        cls, D, tau, hamiltonians, rcutoff=1e-10, degen_ratio=1.0, verbosity=0
    ):
        """
        Initialize simple update at beta = 0 product state.

        Parameters
        ----------
        D : int
            Maximal number of independent multiplets to keep when truncating bonds. For
            abelian symmetries, this is the same as the bond dimension D.
        tau : float
            Imaginary time step.
        hamilts : enumerable of (d**2, d**2) ndarray
            Bond Hamltionians. Must be real symmetric or hermitian.
        rcutoff : float, optional.
            Singular values smaller than cutoff = rcutoff * sv[0] are set to zero to
            improve stability.
        degen_ratio : float
            Consider singular values degenerate if their quotient is above degen_ratio.
            Default is 1.0, exact degeneracy.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        h0 = hamiltonians[0]
        ST = type(h0)
        phys = h0.row_reps[0]
        d = h0.shape[0]
        t0 = np.eye(d).reshape(d, d, 1, 1, 1, 1)

        # quick and dirty. Need singlet as symmetry member.
        if ST.symmetry == "trivial":
            sing = np.array(1)
            t0 = ST.from_array(t0, (phys,), (phys, sing, sing, sing, sing))
        elif ST.symmetry == "U(1)":
            sing = np.array([0], dtype=np.int8)
            t0 = ST.from_array(
                t0, (-phys,), (-phys, sing, sing, sing, sing), conjugate_columns=False
            )
        elif ST.symmetry == "SU(2)":
            sing = np.array([[1], [1]])
            t0 = ST.from_array(t0, (phys,), (phys, sing, sing, sing, sing))

        t1 = t0.group_conjugated()
        #        tA             tB             tC            tD
        #       /  \           /  \           / \           / \
        #      /    \         /    \         /   \         /   \
        #    ////   /\      ///    /\      ///   /\      ///   /\
        #   a234   p  1    a546   p  2    a378  p  1    a687  p  5
        tA = t0.permutate((1, 3, 4, 5), (0, 2))
        tB = t1.permutate((1, 2, 3, 4), (0, 5))
        tC = t1.permutate((1, 2, 3, 5), (0, 4))
        tD = t0.permutate((1, 2, 3, 5), (0, 4))

        return cls(
            D,
            0.0,
            tau,
            rcutoff,
            degen_ratio,
            [tA, tB, tC, tD],
            hamiltonians,
            [np.ones(1)] * cls._n_bonds,
            verbosity,
        )

    def get_tensors(self):
        # adding 1/sqrt(weights) is simpler in dense form
        (sw1, sw2, sw3, sw4, sw5, sw6, sw7, sw8) = [
            1.0 / np.sqrt(w) for w in self.get_weights(sort=False)
        ]
        A0 = self._tensors[0]
        A = np.einsum("ardlpu,u,r,d,l->ardlpu", A0.toarray(), sw1, sw2, sw3, sw4)
        B0 = self._tensors[1]
        B = np.einsum("aurdpl,u,r,d,l->aurdpl", B0.toarray(), sw5, sw4, sw6, sw2)
        C0 = self._tensors[2]
        C = np.einsum("aurlpd,u,r,d,l->aurlpd", C0.toarray(), sw3, sw7, sw1, sw8)
        D0 = self._tensors[3]
        D = np.einsum("aurlpd,u,r,d,l->aurlpd", D0.toarray(), sw6, sw8, sw5, sw7)
        # same problem as in from_infinite_temperature: conjugate_columns has different
        # effect between U(1) and SU(2) from_array.
        cc = self._ST.symmetry == "SU(2)"
        A = self._ST.from_array(A, A0._row_reps, A0._col_reps, conjugate_columns=cc)
        A = A.permutate((4, 0), (5, 1, 2, 3))
        B = self._ST.from_array(B, B0._row_reps, B0._col_reps, conjugate_columns=cc)
        B = B.permutate((4, 0), (1, 2, 3, 5))
        C = self._ST.from_array(C, C0._row_reps, C0._col_reps, conjugate_columns=cc)
        C = C.permutate((4, 0), (1, 2, 5, 3))
        D = self._ST.from_array(D, D0._row_reps, D0._col_reps, conjugate_columns=cc)
        D = D.permutate((4, 0), (1, 2, 5, 3))
        return A, B, C, D

    def get_bond_representations(self):
        A = self._tensors[0]
        D = self._tensors[3]
        r1 = A.conjugate_representation(A.col_reps[1])  # u
        r2 = A.row_reps[1]  # r
        r3 = A.row_reps[2]  # d
        r4 = A.row_reps[3]  # l
        r5 = D.conjugate_representation(D.col_reps[1])  # d
        r6 = D.row_reps[1]  # u
        r7 = D.row_reps[3]  # l
        r8 = D.row_reps[2]  # r
        return r1, r2, r3, r4, r5, r6, r7, r8

    def __repr__(self):
        return f"SimpleUpdate2x2 with {self._symmetry} symmetry and D = {self.D}"

    def evolve(self, beta_evolve):
        """
        Evolve in imaginary time using second order Trotter-Suzuki up to beta.
        Convention: temperature value is the bilayer tensor one, twice the monolayer
        one.
        """
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

        self._update_bond(1, 0, 2, self._gates[0], self._u2u, self._d2d)
        for i in range(niter - 1):  # there is 1 step out of the loop
            self._2nd_order_step_no1()
            self._update_bond(1, 0, 2, self._squared_gates[0], self._r2u, self._u2d)
        self._2nd_order_step_no1()
        self._update_bond(1, 0, 2, self._gates[0], self._r2u, self._u2d)
        self._beta += niter * self._dbeta

    def _2nd_order_step_no1(self):
        # automatically generated list, using updated bonds as input
        self._update_bond(2, 0, 1, self._gates[0], self._u2r, self._l2l)
        self._update_bond(3, 0, 2, self._gates[0], self._r2d, self._d2u)
        self._update_bond(4, 0, 1, self._gates[0], self._d2l, self._l2r)
        self._update_bond(5, 3, 1, self._gates[0], self._d2d, self._r2u)
        self._update_bond(6, 3, 1, self._gates[0], self._d2u, self._u2d)
        self._update_bond(7, 3, 2, self._gates[0], self._u2l, self._u2r)
        self._update_bond(8, 3, 2, self._gates[0], self._l2r, self._r2l)
        self.update_bond_proxy(
            1, 8, 0, 2, 3, self._sqrt_gates[1], self._l2u, self._l2dl, self._r2r, False
        )
        self.update_bond_proxy(
            5, 4, 3, 1, 0, self._sqrt_gates[1], self._r2d, self._d2ur, self._u2l, False
        )
        self.update_bond_proxy(
            7, 1, 3, 2, 0, self._sqrt_gates[1], self._d2l, self._dl2rd, self._l2u, False
        )
        self.update_bond_proxy(
            2, 5, 0, 1, 3, self._sqrt_gates[1], self._u2r, self._ur2lu, self._l2d, False
        )
        self.update_bond_proxy(
            6, 2, 3, 1, 0, self._sqrt_gates[1], self._d2u, self._lu2dl, self._r2r, False
        )
        self.update_bond_proxy(
            3, 7, 0, 2, 3, self._sqrt_gates[1], self._r2d, self._rd2ur, self._u2l, False
        )
        self.update_bond_proxy(
            8, 3, 3, 2, 0, self._sqrt_gates[1], self._l2r, self._ur2lu, self._d2d, False
        )
        self.update_bond_proxy(
            4, 6, 0, 1, 3, self._sqrt_gates[1], self._d2l, self._dl2rd, self._r2u, False
        )
        self.update_bond_proxy(
            5, 7, 1, 3, 2, self._sqrt_gates[1], self._rd2u, self._u2dl, self._lu2r, True
        )
        self.update_bond_proxy(
            1, 2, 2, 0, 1, self._sqrt_gates[1], self._r2d, self._l2ur, self._u2l, True
        )
        self.update_bond_proxy(
            4, 1, 1, 0, 2, self._sqrt_gates[1], self._l2r, self._ur2lu, self._d2d, True
        )
        self.update_bond_proxy(
            8, 5, 2, 3, 1, self._sqrt_gates[1], self._d2l, self._dl2rd, self._r2u, True
        )
        self.update_bond_proxy(
            6, 8, 1, 3, 2, self._sqrt_gates[1], self._u2d, self._rd2ur, self._l2l, True
        )
        self.update_bond_proxy(
            3, 4, 2, 0, 1, self._sqrt_gates[1], self._l2u, self._lu2dl, self._d2r, True
        )
        self.update_bond_proxy(
            2, 3, 1, 0, 2, self._sqrt_gates[1], self._r2l, self._dl2rd, self._u2u, True
        )
        self.update_bond_proxy(
            7, 6, 2, 3, 1, self._gates[1], self._u2r, self._ur2lu, self._l2d, True
        )
        self.update_bond_proxy(
            2, 3, 1, 0, 2, self._sqrt_gates[1], self._d2l, self._rd2rd, self._r2u, True
        )
        self.update_bond_proxy(
            3, 4, 2, 0, 1, self._sqrt_gates[1], self._u2u, self._rd2dl, self._l2r, True
        )
        self.update_bond_proxy(
            6, 8, 1, 3, 2, self._sqrt_gates[1], self._r2d, self._lu2ur, self._u2l, True
        )
        self.update_bond_proxy(
            8, 5, 2, 3, 1, self._sqrt_gates[1], self._l2l, self._ur2rd, self._d2u, True
        )
        self.update_bond_proxy(
            4, 1, 1, 0, 2, self._sqrt_gates[1], self._u2r, self._dl2lu, self._l2d, True
        )
        self.update_bond_proxy(
            1, 2, 2, 0, 1, self._sqrt_gates[1], self._d2d, self._lu2ur, self._r2l, True
        )
        self.update_bond_proxy(
            5, 7, 1, 3, 2, self._sqrt_gates[1], self._l2u, self._rd2dl, self._d2r, True
        )
        self.update_bond_proxy(
            4,
            6,
            0,
            1,
            3,
            self._sqrt_gates[1],
            self._ur2l,
            self._u2rd,
            self._dl2u,
            False,
        )
        self.update_bond_proxy(
            8, 3, 3, 2, 0, self._sqrt_gates[1], self._u2r, self._r2lu, self._l2d, False
        )
        self.update_bond_proxy(
            3, 7, 0, 2, 3, self._sqrt_gates[1], self._d2d, self._lu2ur, self._r2l, False
        )
        self.update_bond_proxy(
            6, 2, 3, 1, 0, self._sqrt_gates[1], self._l2u, self._rd2dl, self._d2r, False
        )
        self.update_bond_proxy(
            2, 5, 0, 1, 3, self._sqrt_gates[1], self._r2r, self._dl2lu, self._u2d, False
        )
        self.update_bond_proxy(
            7, 1, 3, 2, 0, self._sqrt_gates[1], self._d2l, self._ur2rd, self._r2u, False
        )
        self.update_bond_proxy(
            5, 4, 3, 1, 0, self._sqrt_gates[1], self._l2d, self._lu2ur, self._u2l, False
        )
        self.update_bond_proxy(
            1, 8, 0, 2, 3, self._sqrt_gates[1], self._l2u, self._rd2dl, self._d2r, False
        )
        self._update_bond(8, 3, 2, self._gates[0], self._r2r, self._dl2l)
        self._update_bond(7, 3, 2, self._gates[0], self._r2l, self._l2r)
        self._update_bond(6, 3, 1, self._gates[0], self._l2u, self._ur2d)
        self._update_bond(5, 3, 1, self._gates[0], self._u2d, self._d2u)
        self._update_bond(4, 0, 1, self._gates[0], self._u2l, self._u2r)
        self._update_bond(3, 0, 2, self._gates[0], self._l2d, self._r2u)
        self._update_bond(2, 0, 1, self._gates[0], self._d2r, self._r2l)

    def _update_bond(self, bond, iL, iR, gate, lperm, rperm):
        """
        Update bond between tensors iL and iR.
        """
        # bond indices start at 1: -1 shit to get corresponding element in array
        left = self._tensors[iL].permutate(*lperm)
        right = self._tensors[iR].permutate(*rperm)
        nl, nr, nw = self.update_first_neighbor(
            left, right, self._weights[bond - 1], gate
        )

        self._weights[bond - 1] = nw
        self._tensors[iL] = nl
        self._tensors[iR] = nr

    def update_bond_proxy(
        self, bond1, bond2, iL, im, iR, gate, lperm, mperm, rperm, conj
    ):
        """
        Update bonds bond1 and bond2 between tensors iL and iR through tensor im.
        """
        left = self._tensors[iL].permutate(*lperm)
        mid = self._tensors[im].permutate(*mperm)
        right = self._tensors[iR].permutate(*rperm)

        # iL and iR belong to the same sublattice. If it is sublattice B,
        # representations are conjugate. To avoid troubles with weights (which
        # are silently labelled by irreps) and with gates, it is more convenient to
        # conjugate sublattice B to get same representations as on lattice A
        if conj:
            left = left.group_conjugated()
            mid = mid.group_conjugated()
            right = right.group_conjugated()
        left, mid, right, nw1, nw2 = self.update_through_proxy(
            left, mid, right, self._weights[bond1 - 1], self._weights[bond2 - 1], gate
        )

        if conj:  # come back to sublattice B conventions
            left = left.group_conjugated()
            mid = mid.group_conjugated()
            right = right.group_conjugated()
        self._weights[bond1 - 1] = nw1
        self._weights[bond2 - 1] = nw2
        self._tensors[iL] = left
        self._tensors[im] = mid
        self._tensors[iR] = right
