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

    _u2r = ((), ())
    _r2d = ((), ())
    _d2u = ((), ())
    _d2l = ((), ())
    _l2r = ((), ())
    _r2u = ()

    @classmethod
    def from_infinite_temperature(
        cls, Dx, tau, hamiltonians, rcutoff=1e-10, verbosity=0
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
        return cls(
            Dx,
            0.0,
            tau,
            rcutoff,
            [t0, t1, t1.copy(), t0.copy()],
            hamiltonians,
            [sing] * cls._n_bonds,
            [np.ones(1)] * cls._n_bonds,
            verbosity,
        )

    def get_tensors(self):
        # adding 1/sqrt(weights) is simpler in dense form
        (sw1, sw2, sw3, sw4, sw5, sw6, sw7, sw8) = [
            1.0 / np.sqrt(w) for w in self.get_weights(sort=False)
        ]
        A0 = self._tensors[0]
        A = np.einsum("paurdl,u,r,d,l->paurdl", A0.toarray(), sw1, sw2, sw3, sw4)
        B0 = self._tensors[1]
        B = np.einsum("paurdl,u,r,d,l->paurdl", B0.toarray(), sw5, sw4, sw6, sw2)
        C0 = self._tensors[2]
        C = np.einsum("paurdl,u,r,d,l->paurdl", C0.toarray(), sw3, sw7, sw1, sw8)
        D0 = self._tensors[3]
        D = np.einsum("paurdl,u,r,d,l->paurdl", D0.toarray(), sw6, sw8, sw5, sw7)
        # same problem as in from_infinite_temperature: conjugate_columns has differen
        # effect between U(1) and SU(2) from_array.
        cc = self._ST.symmetry == "SU(2)"
        A = self._ST.from_array(A, A0._row_reps, A0._col_reps, conjugate_columns=cc)
        B = self._ST.from_array(B, B0._row_reps, B0._col_reps, conjugate_columns=cc)
        C = self._ST.from_array(C, C0._row_reps, C0._col_reps, conjugate_columns=cc)
        D = self._ST.from_array(D, D0._row_reps, D0._col_reps, conjugate_columns=cc)
        return A, B, C, D

    def get_bond_representations(self):
        A = self._tensors[0]
        D = self._tensors[3]
        r1 = self._ST.conjugate_representation(A.col_reps[1])
        r2 = A.row_reps[1]
        r3 = A.row_reps[2]
        r4 = A.row_reps[3]
        r5 = D.row_reps[1]
        r6 = D.row_reps[2]
        r7 = D.row_reps[3]
        r8 = D.row_reps[3]
        raise NotImplementedError("TODO")
        return r1, r2, r3, r4, r5, r6, r7, r8

    def __repr__(self):
        return f"SimpleUpdate2x2 with {self._symmetry} symmetry"

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

        self._update_bond(1, self._gates[0], self._lperm[0], self._rperm[0])
        for i in range(niter - 1):  # there is 1 step out of the loop
            self._2nd_order_step_no1()
        self._update_bond(1, self._gates[0], self._lperm[6], self._rperm[6])
        self._beta += niter * self._dbeta

        # reset default leg structure
        self._tensors[0] = self._tensors[0].permutate(*self._lperm[7])
        self._tensors[1] = self._tensors[1].permutate(*self._rperm[7])

    def _update_bond(self, bond, iL, iR, gate, lperm, rperm):
        """
        Update bond i between tensors iL and iR.
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

    def update_bond_proxy(self, bond1, bond2, iL, im, iR, gate, lperm, mperm, rperm):
        left = self._tensors[iL].permutate(*lperm)
        mid = self._tensors[im].permutate(*mperm)
        right = self._tensors[iR].permutate(*rperm)
        nl, nm, nr, nw1, nw2 = self.update_through_proxy(
            left, mid, right, self._weights[bond1 - 1], self._weights[bond2 - 1], gate
        )

        self._weights[bond1 - 1] = nw1
        self._weights[bond2 - 1] = nw2
        self._tensors[iL] = nl
        self._tensors[im] = nm
        self._tensors[iR] = nr
