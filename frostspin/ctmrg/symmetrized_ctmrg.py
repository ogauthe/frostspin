from . import rdm
from .abstract_ctmrg import AbstractCTMRG
from .ctm_contract import (
    contract_dl,
    contract_dr,
    contract_ul,
    contract_ur,
)
from .ctm_renormalize import (
    construct_projectors,
    renormalize_quadrant,
    renormalize_T1,
    renormalize_T2,
    renormalize_T3,
    renormalize_T4,
)


class SymmetrizedCTMRG(AbstractCTMRG):
    def construct_enlarged_dr(self, x, y, *, free_memory=False):
        """
        Construct enlarged down right corner by contracting C1, T1, T4 and A. Check _env
        to find an already computed corner, if it does not exist construct it and store
        it in _env.

        Unusual leg ordering: enlarged corners are constructed to be contracted as
        ul-01-ur
        |      |
        1      0
        0      1
        |      |
        dl-10-dr
        meaning enlarged_dr is transposed when compared to standard clockwise order.
        """
        dr = self._env.get_corner_dr(x, y)
        if dr is None:
            dr = contract_dr(
                self._env.get_A(x + 2, y + 2),
                self._env.get_T2(x + 3, y + 2),
                self._env.get_T3(x + 2, y + 3),
                self._env.get_C3(x + 3, y + 3),
            )
            if not free_memory:
                self._env.set_corner_dr(x, y, dr)
                self._env.set_corner_ul(x + 1, y, dr.permute((1, 0, 2), (4, 3, 5)))
        elif free_memory:
            self._env.set_corner_dr(x, y, None)
            self._env.set_corner_ul(x + 1, y, None)
        return dr

    def construct_enlarged_dl(self, x, y, *, free_memory=False):
        """
        Construct enlarged down left corner by contracting C1, T1, T4 and A. Check _env
        to find an already computed corner, if it does not exist construct it and store
        it in _env.
        """
        dl = self._env.get_corner_dl(x, y)
        if dl is None:
            dl = contract_dl(
                self._env.get_T4(x, y + 2),
                self._env.get_A(x + 1, y + 2),
                self._env.get_C4(x, y + 3),
                self._env.get_T3(x + 1, y + 3),
            )
            if not free_memory:
                self._env.set_corner_dl(x, y, dl)
                self._env.set_corner_ur(x + 1, y, dl.permute((1, 0, 2), (4, 3, 5)))
        elif free_memory:
            self._env.set_corner_dl(x, y, None)
            self._env.set_corner_ur(x + 1, y, None)
        return dl

    def construct_enlarged_ul(self, x, y, *, free_memory=False):
        """
        Construct enlarged upper left corner by contracting C1, T1, T4 and A. Check _env
        to find an already computed corner, if it does not exist construct it and store
        it in _env.
        """
        ul = self._env.get_corner_ul(x, y)
        if ul is None:
            ul = contract_ul(
                self._env.get_C1(x, y),
                self._env.get_T1(x + 1, y),
                self._env.get_T4(x, y + 1),
                self._env.get_A(x + 1, y + 1),
            )
            if not free_memory:
                self._env.set_corner_ul(x, y, ul)
                self._env.set_corner_dr(x + 1, y, ul.permute((1, 0, 2), (4, 3, 5)))
        elif free_memory:
            self._env.set_corner_ul(x, y, None)
            self._env.set_corner_dr(x + 1, y, None)
        return ul

    def construct_enlarged_ur(self, x, y, *, free_memory=False):
        """
        Construct enlarged upper right corner by contracting C1, T1, T4 and A. Check
        _env to find an already computed corner, if it does not exist construct it and
        store it in _env.
        """
        ur = self._env.get_corner_ur(x, y)
        if ur is None:
            ur = contract_ur(
                self._env.get_T1(x + 2, y),
                self._env.get_C2(x + 3, y),
                self._env.get_A(x + 2, y + 1),
                self._env.get_T2(x + 3, y + 1),
            )
            if not free_memory:
                self._env.set_corner_ur(x, y, ur)
                self._env.set_corner_dl(x + 1, y, ur.permute((1, 0, 2), (4, 3, 5)))
        elif free_memory:
            self._env.set_corner_ur(x, y, None)
            self._env.set_corner_dl(x, y, None)
        return ur

    def __repr__(self):
        s = f"{self.symmetry()} symmetric SymmetrizedCTMRG with Dmax = {self.Dmax}"
        return s + f" and chi_target = {self.chi_target}"

    def iterate(self):
        if self.verbosity > 1:
            print("Begin CTM iteration")
        self.compute_projectors()
        self.renormalize_tensors()
        if self.verbosity > 1:
            print("Finished CTM iteration")

    def compute_projectors(self):
        """
        Construct all projectors
        """
        trunc = {
            "block_chi_ratio": self.block_chi_ratio,
            "ncv_ratio": self.ncv_ratio,
            "rtol": self.cutoff,
            "degen_ratio": self.degen_ratio,
        }

        dl = self.construct_enlarged_dl(0, 0)
        dr = self.construct_enlarged_dr(0, 0)
        ur = self.construct_enlarged_ur(0, 0)
        ul = self.construct_enlarged_ul(0, 0)

        old_c2 = self._env.get_C2(0, 1)
        P, Pt = construct_projectors(dr, ur, ul, dl, self.chi_target, old_c2, **trunc)
        self._env.store_up_projectors(0, 0, P, Pt)
        self._env.store_down_projectors(
            0, 1, P.permute((1, 0, 2), (3,)), Pt.permute((1, 0, 2), (3,))
        )

        old_c3 = self._env.get_C3(0, 0).transpose()
        P, Pt = construct_projectors(dl, dr, ur, ul, self.chi_target, old_c3, **trunc)
        self._env.store_right_projectors(1, 0, P, Pt)
        self._env.store_left_projectors(
            0, 0, P.permute((1, 0, 2), (3,)), Pt.permute((1, 0, 2), (3,))
        )

        old_c4 = self._env.get_C4(1, 0)
        P, Pt = construct_projectors(ul, dl, dr, ur, self.chi_target, old_c4, **trunc)
        self._env.store_down_projectors(1, 1, P, Pt)
        self._env.store_up_projectors(
            0, 1, P.permute((1, 0, 2), (3,)), Pt.permute((1, 0, 2), (3,))
        )

        old_c1 = self._env.get_C1(1, 1)
        P, Pt = construct_projectors(ur, ul, dl, dr, self.chi_target, old_c1, **trunc)
        self._env.store_left_projectors(0, 1, P, Pt)
        self._env.store_right_projectors(
            0, 0, P.permute((1, 0, 2), (3,)), Pt.permute((1, 0, 2), (3,))
        )

    def renormalize_tensors(self):
        """
        Renormalize all tensors
        """
        A = self._env.get_A(0, 0)
        nC1 = renormalize_quadrant(
            self._env.get_up_P(1, 1),
            self.construct_enlarged_ul(1, 1),
            self._env.get_left_Pt(1, 0),
        )
        nT1 = renormalize_T1(
            self._env.get_up_Pt(0, 1),
            self._env.get_T1(0, 1),
            A,
            self._env.get_up_P(1, 1),
        )
        nC2 = renormalize_quadrant(
            self._env.get_right_P(1, 1),
            self.construct_enlarged_ur(0, 1),
            self._env.get_up_Pt(0, 1),
        )
        nT2 = renormalize_T2(
            self._env.get_right_Pt(1, 0),
            self._env.get_T2(1, 0),
            A,
            self._env.get_right_P(1, 1),
        )
        nC3 = renormalize_quadrant(
            self._env.get_down_P(1, 1),
            self.construct_enlarged_dr(0, 0),
            self._env.get_right_Pt(1, 0),
        ).transpose()
        nT3 = renormalize_T3(
            self._env.get_down_Pt(0, 1),
            self._env.get_T3(0, 1),
            A,
            self._env.get_down_P(1, 1),
        )
        nC4 = renormalize_quadrant(
            self._env.get_left_P(1, 1),
            self.construct_enlarged_dl(1, 0),
            self._env.get_down_Pt(0, 1),
        )
        nT4 = renormalize_T4(
            self._env.get_left_Pt(1, 0),
            self._env.get_T4(1, 0),
            A,
            self._env.get_left_P(1, 1),
        )

        self._env = type(self._env)(
            self.cell,
            [A, self._env.get_A(0, 1)],
            [nC1, nC3.transpose()],
            [nC2, nC4],
            [nC3, nC1.transpose()],
            [nC4, nC2],
            [nT1, nT3.permute((3,), (1, 0, 2))],
            [nT2, nT4.permute((3,), (0, 2, 1))],
            [nT3, nT1.permute((2, 1, 3), (0,))],
            [nT4, nT2.permute((1,), (3, 2, 0))],
        )

    def compute_rdm1x2(self, x, y):
        if self.verbosity > 1:
            print(f"Compute rdm 1x2 with C1 coord = ({x},{y})")
        return rdm.rdm_symmetrized_1x2(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_C4(x, y + 2),
            self._env.get_T3(x + 1, y + 2),
        )

    def compute_rdm2x1(self, x, y):
        if self.verbosity > 1:
            print(f"Compute rdm 2x1 with C1 coord = ({x},{y})")
        return rdm.rdm_symmetrized_2x1(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_C2(x + 2, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_T2(x + 2, y + 1),
        )

    def compute_rdm_2nd_neighbor_cell(self, *, free_memory=True):
        """
        Compute reduced density matrix for every couple of inquivalent cell next nearest
        neighbor sites.
        """
        # Very heavy in memory. Expected to be called only after convergence, so
        # corners will not be needed for next iterations, drop them to save memory
        if free_memory:
            self._env.reset_constructed_corners()
        if self.verbosity > 1:
            print("Compute rdm for every cell next nearest neighbor sites")
        rdm_dr = self.compute_rdm_diag_dr(0, 0, free_memory=free_memory)
        rdm_ur = self.compute_rdm_diag_ur(0, 0, free_memory=free_memory)

        return [rdm_dr, rdm_dr.transpose()], [rdm_ur, rdm_ur.transpose()]
