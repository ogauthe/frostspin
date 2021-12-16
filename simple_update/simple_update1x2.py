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

    _lperm = (
        ((1, 3, 4, 5), (0, 2)),
        ((0, 5, 2, 3), (4, 1)),
        ((0, 1, 5, 3), (4, 2)),
        ((0, 1, 2, 5), (4, 3)),
        ((0, 1, 2, 5), (4, 3)),
        ((0, 1, 5, 3), (4, 2)),
        ((0, 5, 2, 3), (4, 1)),
        ((4, 0), (5, 1, 2, 3)),
    )

    _rperm = (
        ((4, 0), (1, 5, 2, 3)),
        ((3, 1), (2, 0, 4, 5)),
        ((4, 1), (2, 3, 0, 5)),
        ((5, 1), (2, 3, 4, 0)),
        ((5, 1), (2, 3, 4, 0)),
        ((4, 1), (2, 3, 0, 5)),
        ((3, 1), (2, 0, 4, 5)),
        ((1, 2), (4, 5, 0, 3)),
    )

    @classmethod
    def from_infinite_temperature(
        cls, Dx, tau, hamiltonians, rcutoff=1e-11, verbosity=0
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

        # tensors A and B start from the default state
        #      A,B
        #     /   \
        #    /\   /\\\
        #   p a   1 234
        #
        # cast them to structure required for update 1, then before each update
        # permutate them to the next one.

        # for tensor A, this means start from ((p,a), (1,2,3,4))
        # -> ((a,2,3,4), (p,1))     swap = ((1,3,4,5), (0,2))
        # -> ((a,1,3,4), (p,2))     swap = ((0,5,2,3), (4,1))
        # -> ((a,1,2,4), (p,3))     swap = ((0,1,5,3), (4,2))
        # -> ((a,1,2,3), (p,4))     swap = ((0,1,2,5), (4,3))
        # -> ((a,1,2,4), (p,3))     swap = ((0,1,2,5), (4,3))
        # -> ((a,1,3,4), (p,2))     swap = ((0,1,5,3), (4,2))
        # -> ((a,2,3,4), (p,1))     swap = ((0,5,2,3), (4,1))
        # ...
        # -> ((p,a), (1,2,3,4))     swap = ((4,0), (5,1,2,3))

        # for tensor B, this means start from ((p,a), (3,4,1,2))
        # -> ((1,p), (3,4,p,1))     swap = ((4,0), (1,5,2,3))
        # -> ((2,p), (3,4,p,2))     swap = ((3,1), (2,0,4,5))
        # -> ((3,p), (2,4,p,3))     swap = ((4,1), (2,3,0,5))
        # -> ((4,p), (2,3,p,4))     swap = ((5,1), (2,3,4,0))
        # -> ((3,p), (2,4,p,3))     swap = ((5,1), (2,3,4,0))
        # -> ((2,p), (3,4,p,2))     swap = ((4,1), (2,3,0,5))
        # -> ((1,p), (3,4,p,1))     swap = ((3,1), (2,0,4,5))
        # ...
        # -> ((p,a), (3,4,1,2))     swap = ((1,2), (4,5,0,3))

        self._update_bond(1, self._gates[0], self._lperm[0], self._rperm[0])
        for i in range(niter - 1):  # there is 1 step out of the loop
            self._update_bond(2, self._gates[0], self._lperm[1], self._rperm[1])
            self._update_bond(3, self._gates[0], self._lperm[2], self._rperm[2])
            self._update_bond(4, self._squared_gates[0], self._lperm[3], self._rperm[3])
            self._update_bond(3, self._gates[0], self._lperm[4], self._rperm[4])
            self._update_bond(2, self._gates[0], self._lperm[5], self._rperm[5])
            self._update_bond(1, self._squared_gates[0], self._lperm[6], self._rperm[6])
            self._beta += self._dbeta
        self._update_bond(2, self._gates[0], self._lperm[1], self._rperm[1])
        self._update_bond(3, self._gates[0], self._lperm[2], self._rperm[2])
        self._update_bond(4, self._gates[0], self._lperm[3], self._rperm[3])
        self._update_bond(3, self._gates[0], self._lperm[4], self._rperm[4])
        self._update_bond(2, self._gates[0], self._lperm[5], self._rperm[5])
        self._update_bond(1, self._gates[0], self._lperm[6], self._rperm[6])
        self._beta += self._dbeta

        # reset default leg structure
        self._tensors[0].permutate(*self._lperm[7])
        self._tensors[1].permutate(*self._rperm[7])

    def _update_bond(self, i, gate, lperm, rperm):
        """
        Update bond i between tensors A and B.
        Tranpose tensors A and B, assuming default leg ordering.
        """
        # bond indices start at 1: -1 shit to get corresponding element in array
        if self.verbosity > 2:
            print(
                f"update bond {i}: rep {i} = {self._bond_representations[i - 1]},",
            )

        left = self._tensors[0].permutate(*lperm)
        right = self._tensors[1].permutate(*rperm)
        nl, nr, nw = self.update_first_neighbor(left, right, self._weights[i - 1], gate)

        self._bond_representations[i - 1] = left.col_reps[1]
        self._weights[i - 1] = nw
        self._tensors[0] = nl
        self._tensors[1] = nr
