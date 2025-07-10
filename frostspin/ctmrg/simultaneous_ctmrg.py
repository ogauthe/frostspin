import numpy as np

from .abstract_ctmrg import AbstractCTMRG
from .ctm_renormalize import (
    construct_projectors,
    renormalize_quadrant,
    renormalize_T1,
    renormalize_T2,
    renormalize_T3,
    renormalize_T4,
)


class SimultaneousCTMRG(AbstractCTMRG):

    def iterate(self):
        if self.verbosity > 1:
            print("Begin CTM iteration")
        self.construct_enlarged_corners()
        self.compute_projectors()
        self.renormalize_tensors()
        if self.verbosity > 1:
            print("Finished CTM iteration")

    def construct_enlarged_corners(self):
        """
        Construct all enlarged corners
        """
        for x, y in self._site_coords:
            self.construct_enlarged_dr(x, y)
            self.construct_enlarged_ur(x, y)
            self.construct_enlarged_ul(x, y)
            self.construct_enlarged_dl(x, y)

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

        for x, y in self._site_coords:
            dl = self.construct_enlarged_dl(x, y)
            dr = self.construct_enlarged_dr(x, y)
            ur = self.construct_enlarged_ur(x, y)
            ul = self.construct_enlarged_ul(x, y)

            old_c2 = self._env.get_C2(x + 2, y + 1)
            P, Pt = construct_projectors(
                dr, ur, ul, dl, self.chi_target, old_c2, **trunc
            )
            self._env.store_up_projectors(x + 2, y, P, Pt)

            old_c3 = self._env.get_C3(x + 2, y + 2).transpose()
            P, Pt = construct_projectors(
                dl, dr, ur, ul, self.chi_target, old_c3, **trunc
            )
            self._env.store_right_projectors(x + 3, y + 2, P, Pt)

            old_c4 = self._env.get_C4(x + 1, y + 2)
            P, Pt = construct_projectors(
                ul, dl, dr, ur, self.chi_target, old_c4, **trunc
            )
            self._env.store_down_projectors(x + 1, y + 3, P, Pt)

            old_c1 = self._env.get_C1(x + 1, y + 1)
            P, Pt = construct_projectors(
                ur, ul, dl, dr, self.chi_target, old_c1, **trunc
            )
            self._env.store_left_projectors(x, y + 1, P, Pt)

    def renormalize_tensors(self):
        """
        Renormalize all tensors
        """
        new_env_tensors = np.empty((self.n_sites, 9), dtype=object)
        for i, (x, y) in enumerate(self._site_coords):
            A = self._env.get_A(x, y)
            nC1 = renormalize_quadrant(
                self._env.get_up_P(x + 1, y - 1),
                self.construct_enlarged_ul(x - 1, y - 1),
                self._env.get_left_Pt(x - 1, y),
            )
            nT1 = renormalize_T1(
                self._env.get_up_Pt(x, y - 1),
                self._env.get_T1(x, y - 1),
                A,
                self._env.get_up_P(x + 1, y - 1),
            )
            nC2 = renormalize_quadrant(
                self._env.get_right_P(x + 1, y + 1),
                self.construct_enlarged_ur(x - 2, y - 1),
                self._env.get_up_Pt(x, y - 1),
            )
            nT2 = renormalize_T2(
                self._env.get_right_Pt(x + 1, y),
                self._env.get_T2(x + 1, y),
                A,
                self._env.get_right_P(x + 1, y + 1),
            )
            nC3 = renormalize_quadrant(
                self._env.get_down_P(x - 1, y + 1),
                self.construct_enlarged_dr(x - 2, y - 2),
                self._env.get_right_Pt(x + 1, y),
            ).transpose()
            nT3 = renormalize_T3(
                self._env.get_down_Pt(x, y + 1),
                self._env.get_T3(x, y + 1),
                A,
                self._env.get_down_P(x - 1, y + 1),
            )
            nC4 = renormalize_quadrant(
                self._env.get_left_P(x - 1, y - 1),
                self.construct_enlarged_dl(x - 1, y - 2),
                self._env.get_down_Pt(x, y + 1),
            )
            nT4 = renormalize_T4(
                self._env.get_left_Pt(x - 1, y),
                self._env.get_T4(x - 1, y),
                A,
                self._env.get_left_P(x - 1, y - 1),
            )
            new_env_tensors[i] = A, nC1, nC2, nC3, nC4, nT1, nT2, nT3, nT4

        self._env = type(self._env)(self.cell, *new_env_tensors.T.tolist())
