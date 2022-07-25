import numpy as np

from simple_update.simple_update import SimpleUpdate


class SimpleUpdate1x2(SimpleUpdate):
    r"""
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

    Out of evolve method, leg structure is:
            A                   B
           /  \                /  \
          /    \              /    \
        ////   /\           ///    /\
       a234   p  1         a234   p  1
    """

    _unit_cell = "AB"
    _classname = "SimpleUpdate1x2"  # used in save/load to check consistency
    _n_bonds = 4
    _n_hamiltonians = 1
    _n_tensors = 2

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
        phys = h0.row_reps[0]
        d = h0.shape[0]
        t0 = np.eye(d).reshape(d, 1, 1, 1, d, 1)

        # for simplicity, impose all Hamiltonians to have standard signature
        s = np.array([False, False, True, True])
        if not all((h.signature == s).all() for h in hamiltonians):
            raise ValueError(f"Hamiltonians must have signature {s}")

        # use same signature for physical and ancilla legs on all tensors:
        # True for physical
        # False for ancilla
        # for easy contraction with Hamiltonian
        # however virtual legs need to have opposite signatures on 2 sublattices
        # choose signature so that weights are always applied on the side where they
        # were cut: right for A, left for B.

        # left and right carry the same representations.
        # use same signature for physical and ancilla legs
        # however virtual legs need to have opposite signatures
        #
        #       left                right
        #       /  \                /  \
        #      /    \              /    \
        #    ////   /\           ///    /\
        #   a234   p  1         a234   p  1
        #   -+++   +  +         ----   +  -
        sing = h0.singlet
        sl = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
        left = h0.from_array(t0, (phys, sing, sing, sing), (phys, sing), signature=sl)
        sr = np.array([0, 0, 0, 0, 1, 0], dtype=bool)
        right = h0.from_array(t0, (phys, sing, sing, sing), (phys, sing), signature=sr)
        return cls(
            D,
            0.0,
            tau,
            rcutoff,
            degen_ratio,
            [left, right],
            hamiltonians,
            [[np.ones((1,))] for i in range(cls._n_bonds)],
            verbosity,
        )

    def get_bond_representations(self):
        left = self._tensors[0]
        r1 = left.col_reps[1]
        r2 = left.row_reps[1]
        r3 = left.row_reps[2]
        r4 = left.row_reps[3]
        return r1, r2, r3, r4

    def get_tensors(self):
        # adding 1/sqrt(weights) is simpler in dense form
        sw1, sw2, sw3, sw4 = [1.0 / np.sqrt(w) for w in self.get_weights(sort=False)]
        A0 = self._tensors[0]
        A = np.einsum("ardlpu,u,r,d,l->ardlpu", A0.toarray(), sw1, sw2, sw3, sw4)
        B0 = self._tensors[1]
        B = np.einsum("alurpd,u,r,d,l->alurpd", B0.toarray(), sw3, sw4, sw1, sw2)
        A = self._ST.from_array(A, A0._row_reps, A0._col_reps, signature=A0.signature)
        B = self._ST.from_array(B, B0._row_reps, B0._col_reps, signature=B0.signature)
        A = A.permutate((4, 0), (5, 1, 2, 3))
        B = B.permutate((4, 0), (2, 3, 5, 1))
        return A, B

    def __repr__(self):
        return f"SimpleUpdate1x2 with {self._symmetry} symmetry and D = {self.D}"

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
        left = self._tensors[0].permutate(*swap)
        right = self._tensors[1].permutate(*swap)
        nl, nr, nw = self.update_first_neighbor(left, right, self._weights[i - 1], gate)

        self._weights[i - 1] = nw
        self._tensors[0] = nl
        self._tensors[1] = nr
