from simple_update.simple_update2x2 import SimpleUpdate2x2


class SimpleUpdateMonostripe(SimpleUpdate2x2):
    """
    Simple update algorithm on plaquette AB//CD with stripes in one direction A->D /
    B -> C only.
    Very similar to parent class SimpleUpdate2x2, just change _2nd_order_step_no1 to
    remove half of the second neighbor bonds.
    """

    _classname = "SimpleUpdateMonostripe"

    # permutations used in evolve (generated)
    _d2d = ((0, 1, 2, 3), (4, 5))
    _d2l = ((0, 1, 2, 5), (4, 3))
    _d2r = ((0, 1, 5, 3), (4, 2))
    _d2u = ((0, 2, 5, 3), (4, 1))
    _d2ur = ((4, 0, 5, 3), (1, 2))
    _dl2dl = ((0, 1, 2, 3), (4, 5))
    _dl2l = ((1, 2, 3, 4), (0, 5))
    _dl2rd = ((0, 1, 2, 5), (3, 4))
    _dl2u = ((1, 3, 4, 5), (0, 2))
    _dl2ur = ((0, 1, 4, 5), (2, 3))
    _l2d = ((0, 1, 2, 5), (4, 3))
    _l2dl = ((4, 0, 1, 2), (3, 5))
    _l2l = ((0, 1, 2, 3), (4, 5))
    _l2r = ((0, 1, 3, 5), (4, 2))
    _l2u = ((0, 2, 3, 5), (4, 1))
    _l2ur = ((4, 0, 3, 5), (1, 2))
    _lu2r = ((1, 5, 3, 4), (0, 2))
    _lu2ur = ((0, 1, 3, 4), (5, 2))
    _r2d = ((0, 1, 5, 3), (4, 2))
    _r2l = ((0, 1, 5, 2), (4, 3))
    _r2lu = ((4, 0, 5, 2), (3, 1))
    _r2r = ((0, 1, 2, 3), (4, 5))
    _r2u = ((0, 5, 2, 3), (4, 1))
    _rd2dl = ((0, 1, 2, 4), (5, 3))
    _rd2u = ((1, 4, 5, 3), (0, 2))
    _u2d = ((0, 5, 1, 3), (4, 2))
    _u2dl = ((4, 0, 5, 1), (2, 3))
    _u2l = ((0, 5, 1, 2), (4, 3))
    _u2r = ((0, 5, 2, 3), (4, 1))
    _u2rd = ((4, 0, 5, 3), (1, 2))
    _u2u = ((0, 1, 2, 3), (4, 5))
    _ur2d = ((1, 4, 5, 3), (0, 2))
    _ur2dl = ((0, 1, 4, 5), (2, 3))
    _ur2l = ((1, 4, 5, 2), (0, 3))
    _ur2lu = ((0, 1, 5, 2), (3, 4))

    def __repr__(self):
        s = f"SimpleUpdateMonostripe with {self._symmetry} symmetry and D = {self.D}"
        return s

    def _2nd_order_step_no1(self):
        # automatically generated list, using updated bonds as input
        self._update_bond(2, 0, 1, self._gates[0], self._u2r, self._l2l)
        self._update_bond(3, 0, 2, self._gates[0], self._r2d, self._d2u)
        self._update_bond(4, 0, 1, self._gates[0], self._d2l, self._l2r)
        self._update_bond(5, 3, 1, self._gates[0], self._d2d, self._r2u)
        self._update_bond(6, 3, 1, self._gates[0], self._d2u, self._u2d)
        self._update_bond(7, 3, 2, self._gates[0], self._u2l, self._u2r)
        self._update_bond(8, 3, 2, self._gates[0], self._l2r, self._r2l)

        sg1 = self._sqrt_gates[1]
        self.update_bond_proxy(1, 8, 0, 2, 3, sg1, self._l2u, self._l2dl, self._r2r)
        self.update_bond_proxy(5, 4, 3, 1, 0, sg1, self._r2d, self._d2ur, self._u2l)
        self.update_bond_proxy(6, 2, 3, 1, 0, sg1, self._d2u, self._ur2dl, self._l2r)
        self.update_bond_proxy(3, 7, 0, 2, 3, sg1, self._r2d, self._dl2ur, self._u2l)
        self.update_bond_proxy(8, 3, 3, 2, 0, sg1, self._l2r, self._ur2lu, self._d2d)
        self.update_bond_proxy(4, 6, 0, 1, 3, sg1, self._d2l, self._dl2rd, self._r2u)
        self.update_bond_proxy(5, 7, 1, 3, 2, sg1, self._rd2u, self._u2dl, self._lu2r)
        self.update_bond_proxy(
            1, 2, 2, 0, 1, self._gates[1], self._r2d, self._l2ur, self._u2l
        )
        self.update_bond_proxy(5, 7, 1, 3, 2, sg1, self._l2u, self._dl2dl, self._d2r)
        self.update_bond_proxy(4, 6, 0, 1, 3, sg1, self._ur2l, self._u2rd, self._dl2u)
        self.update_bond_proxy(8, 3, 3, 2, 0, sg1, self._u2r, self._r2lu, self._l2d)
        self.update_bond_proxy(3, 7, 0, 2, 3, sg1, self._d2d, self._lu2ur, self._r2l)
        self.update_bond_proxy(6, 2, 3, 1, 0, sg1, self._l2u, self._rd2dl, self._d2r)
        self.update_bond_proxy(5, 4, 3, 1, 0, sg1, self._u2d, self._dl2ur, self._r2l)
        self.update_bond_proxy(1, 8, 0, 2, 3, sg1, self._l2u, self._ur2dl, self._d2r)

        self._update_bond(8, 3, 2, self._gates[0], self._r2r, self._dl2l)
        self._update_bond(7, 3, 2, self._gates[0], self._r2l, self._l2r)
        self._update_bond(6, 3, 1, self._gates[0], self._l2u, self._ur2d)
        self._update_bond(5, 3, 1, self._gates[0], self._u2d, self._d2u)
        self._update_bond(4, 0, 1, self._gates[0], self._u2l, self._u2r)
        self._update_bond(3, 0, 2, self._gates[0], self._l2d, self._r2u)
        self._update_bond(2, 0, 1, self._gates[0], self._d2r, self._r2l)
