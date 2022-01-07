import numpy as np

from simple_update.simple_update import SimpleUpdate


class SimpleUpdate1x2(SimpleUpdate):
    """
    Simple update algorithm on plaquette AB, with
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
    _n_hamiltonians = 1
    _n_tensors = 2

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
            left = ST.from_array(t0, (phys,), (phys, sing, sing, sing, sing))
        elif ST.symmetry == "U(1)":
            sing = np.array([0], dtype=np.int8)
            left = ST.from_array(
                t0, (-phys,), (-phys, sing, sing, sing, sing), conjugate_columns=False
            )
        elif ST.symmetry == "SU(2)":
            sing = np.array([[1], [1]])
            left = ST.from_array(t0, (phys,), (phys, sing, sing, sing, sing))

        left = left.permutate((1, 3, 4, 5), (0, 2))
        #       left                right
        #       /  \                /  \
        #      /    \              /    \
        #    ////   /\           ///    /\
        #   a234   p  1         a234   p  1
        return cls(
            Dx,
            0.0,
            tau,
            rcutoff,
            [left, left.group_conjugated().copy()],
            hamiltonians,
            [sing] * cls._n_bonds,
            [np.ones(1)] * cls._n_bonds,
            verbosity,
        )

    def get_tensors(self):
        # adding 1/sqrt(weights) is simpler in dense form
        sw1, sw2, sw3, sw4 = [1.0 / np.sqrt(w) for w in self.get_weights(sort=False)]
        A0 = self._tensors[0]
        A = np.einsum("ardlpu,u,r,d,l->ardlpu", A0.toarray(), sw1, sw2, sw3, sw4)
        B0 = self._tensors[1]
        B = np.einsum("alurpd,u,r,d,l->alurpd", B0.toarray(), sw3, sw4, sw1, sw2)
        # same problem as in from_infinite_temperature: conjugate_columns has differen
        # effect between U(1) and SU(2) from_array.
        cc = self._ST.symmetry == "SU(2)"
        A = self._ST.from_array(A, A0._row_reps, A0._col_reps, conjugate_columns=cc)
        B = self._ST.from_array(B, B0._row_reps, B0._col_reps, conjugate_columns=cc)
        A = A.permutate((4, 0), (5, 1, 2, 3))
        B = B.permutate((4, 0), (2, 3, 5, 1))
        return A, B

    def __repr__(self):
        return f"SimpleUpdate1x2 with {self._symmetry} symmetry"

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

        # tensors A and B start from the default state corresponding to update 1
        #      A,B
        #     /   \
        #   ///   /\
        #  a234  p  1
        #
        # before each update, permutate them to required structure
        # note that they always have the same structure

        # this means start from ((a, 2, 3, 4), (p, 1))
        # -> ((a,2,3,4), (p,1))     swap = ((0,1,2,3), (4,5)) = nothing
        # -> ((a,1,3,4), (p,2))     swap = ((0,5,2,3), (4,1))
        # -> ((a,1,2,4), (p,3))     swap = ((0,1,5,3), (4,2))
        # -> ((a,1,2,3), (p,4))     swap = ((0,1,2,5), (4,3))
        # -> ((a,1,2,4), (p,3))     swap = ((0,1,2,5), (4,3))
        # -> ((a,1,3,4), (p,2))     swap = ((0,1,5,3), (4,2))
        # -> ((a,2,3,4), (p,1))     swap = ((0,5,2,3), (4,1))

        self._update_bond(1, self._gates[0], ((0, 1, 2, 3), (4, 5)))
        for i in range(niter - 1):  # there is 1 step out of the loop
            self._update_bond(2, self._gates[0], ((0, 5, 2, 3), (4, 1)))
            self._update_bond(3, self._gates[0], ((0, 1, 5, 3), (4, 2)))
            self._update_bond(4, self._squared_gates[0], ((0, 1, 2, 5), (4, 3)))
            self._update_bond(3, self._gates[0], ((0, 1, 2, 5), (4, 3)))
            self._update_bond(2, self._gates[0], ((0, 1, 5, 3), (4, 2)))
            self._update_bond(1, self._squared_gates[0], ((0, 5, 2, 3), (4, 1)))
        self._update_bond(2, self._gates[0], ((0, 5, 2, 3), (4, 1)))
        self._update_bond(3, self._gates[0], ((0, 1, 5, 3), (4, 2)))
        self._update_bond(4, self._gates[0], ((0, 1, 2, 5), (4, 3)))
        self._update_bond(3, self._gates[0], ((0, 1, 2, 5), (4, 3)))
        self._update_bond(2, self._gates[0], ((0, 1, 5, 3), (4, 2)))
        self._update_bond(1, self._gates[0], ((0, 5, 2, 3), (4, 1)))
        self._beta += niter * self._dbeta

    def _update_bond(self, i, gate, swap):
        """
        Update bond i between tensors A and B.
        """
        # bond indices start at 1: -1 shit to get corresponding element in array
        if self.verbosity > 2:
            print(
                f"update bond {i}: rep {i} = {self._bond_representations[i - 1]},",
            )

        left = self._tensors[0].permutate(*swap)
        right = self._tensors[1].permutate(*swap)
        nl, nr, nw = self.update_first_neighbor(left, right, self._weights[i - 1], gate)

        self._bond_representations[i - 1] = left.col_reps[1]
        self._weights[i - 1] = nw
        self._tensors[0] = nl
        self._tensors[1] = nr
