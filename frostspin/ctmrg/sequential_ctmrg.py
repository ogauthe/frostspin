from .abstract_ctmrg import AbstractCTMRG
from .ctm_renormalize import (
    construct_projectors,
    renormalize_C1_left,
    renormalize_C1_up,
    renormalize_C2_right,
    renormalize_C2_up,
    renormalize_C3_down,
    renormalize_C3_right,
    renormalize_C4_down,
    renormalize_C4_left,
    renormalize_T1,
    renormalize_T2,
    renormalize_T3,
    renormalize_T4,
)


class SequentialCTMRG(AbstractCTMRG):

    def __repr__(self):
        s = f"{self.symmetry()} symmetric SequentialCTMRG with Dmax = {self.Dmax}"
        return s + f" and chi_target = {self.chi_target}"

    def iterate(self):
        if self.verbosity > 1:
            print("Begin CTM iteration")
        self.up_move()
        self.right_move()
        self.down_move()
        self.left_move()
        if self.verbosity > 1:
            print("Finished CTM iteration")

    def up_move(self):
        """
        Absorb a row on the upper side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete ul
        and ur corners which have been renormalized.
        """
        # 1) compute isometries for every non-equivalent sites
        # construct corners, free memory as soon as possible for corners that will be
        # updated by this move.
        # use last renornalized corner to estimate block sizes in each symmetry sector
        # and reduce total number of computed singular vecotrs
        # Note that last renormalized corner coordinates correspond to the new corner,
        # obtain from the isometries under construction.
        for x, y in self._site_coords:
            P, Pt = construct_projectors(
                self.construct_enlarged_dr(x, y),
                self.construct_enlarged_ur(x, y, free_memory=True),
                self.construct_enlarged_ul(x, y, free_memory=True),
                self.construct_enlarged_dl(x, y),
                self.chi_target,
                self._env.get_C2(x + 2, y + 1),
                block_chi_ratio=self.block_chi_ratio,
                ncv_ratio=self.ncv_ratio,
                rtol=self.cutoff,
                degen_ratio=self.degen_ratio,
            )
            self._env.store_up_projectors(x + 2, y, P, Pt)

        # 2) renormalize every non-equivalent C1, T1 and C2
        # need all projectors to be constructed at this time
        for x, y in self._site_coords:
            P = self._env.get_up_P(x + 1, y)
            Pt = self._env.get_up_Pt(x, y)
            nC1 = renormalize_C1_up(
                self._env.get_C1(x, y), self._env.get_T4(x, y + 1), P
            )

            A = self._env.get_A(x, y + 1)
            nT1 = renormalize_T1(Pt, self._env.get_T1(x, y), A, P)

            nC2 = renormalize_C2_up(
                self._env.get_C2(x, y), self._env.get_T2(x, y + 1), Pt
            )
            self._env.store_renormalized_tensors(x, y + 1, nC1, nT1, nC2)

        # 3) store renormalized tensors in the environment
        # renormalization reads C1[x,y] but writes C1[x,y+1]
        # => need to compute every renormalized tensors before storing any of them
        self._env.set_renormalized_tensors_up()

    def right_move(self):
        """
        Absorb a column on the right side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete ur
        and dr corners which have been renormalized.
        """
        if self.verbosity > 1:
            print("\nstart right move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._site_coords:
            P, Pt = construct_projectors(
                self.construct_enlarged_dl(x, y),
                self.construct_enlarged_dr(x, y, free_memory=True),
                self.construct_enlarged_ur(x, y, free_memory=True),
                self.construct_enlarged_ul(x, y),
                self.chi_target,
                self._env.get_C3(x + 2, y + 2).transpose(),
                block_chi_ratio=self.block_chi_ratio,
                ncv_ratio=self.ncv_ratio,
                rtol=self.cutoff,
                degen_ratio=self.degen_ratio,
            )
            self._env.store_right_projectors(x + 3, y + 2, P, Pt)

        # 2) renormalize tensors by absorbing column
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._site_coords:
            P = self._env.get_right_P(x, y + 1)
            Pt = self._env.get_right_Pt(x, y)
            nC2 = renormalize_C2_right(
                self._env.get_C2(x, y), self._env.get_T1(x - 1, y), P
            )

            A = self._env.get_A(x - 1, y)
            nT2 = renormalize_T2(Pt, self._env.get_T2(x, y), A, P)

            nC3 = renormalize_C3_right(
                self._env.get_C3(x, y), self._env.get_T3(x - 1, y), Pt
            )
            self._env.store_renormalized_tensors(x - 1, y, nC2, nT2, nC3)

        # 3) store renormalized tensors in the environment
        self._env.set_renormalized_tensors_right()
        if self.verbosity > 1:
            print("right move completed")

    def down_move(self):
        """
        Absorb a row on the down side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete dl
        and dr corners which have been renormalized.
        """
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._site_coords:
            P, Pt = construct_projectors(
                self.construct_enlarged_ul(x, y),
                self.construct_enlarged_dl(x, y, free_memory=True),
                self.construct_enlarged_dr(x, y, free_memory=True),
                self.construct_enlarged_ur(x, y),
                self.chi_target,
                self._env.get_C4(x + 1, y + 2),
                block_chi_ratio=self.block_chi_ratio,
                ncv_ratio=self.ncv_ratio,
                rtol=self.cutoff,
                degen_ratio=self.degen_ratio,
            )
            self._env.store_down_projectors(x + 1, y + 3, P, Pt)

        # 2) renormalize every non-equivalent C3, T3 and C4
        for x, y in self._site_coords:
            P = self._env.get_down_P(x - 1, y)
            Pt = self._env.get_down_Pt(x, y)
            nC3 = renormalize_C3_down(
                self._env.get_C3(x, y), self._env.get_T2(x, y - 1), P
            )

            A = self._env.get_A(x, y - 1)
            nT3 = renormalize_T3(Pt, self._env.get_T3(x, y), A, P)

            nC4 = renormalize_C4_down(
                self._env.get_C4(x, y), self._env.get_T4(x, y - 1), Pt
            )
            self._env.store_renormalized_tensors(x, y - 1, nC3, nT3, nC4)

        # 3) store renormalized tensors in the environment
        self._env.set_renormalized_tensors_down()

    def left_move(self):
        """
        Absorb a column on the left side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete ul
        and dl corners which have been renormalized.
        """
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._site_coords:
            P, Pt = construct_projectors(
                self.construct_enlarged_ur(x, y),
                self.construct_enlarged_ul(x, y, free_memory=True),
                self.construct_enlarged_dl(x, y, free_memory=True),
                self.construct_enlarged_dr(x, y),
                self.chi_target,
                self._env.get_C1(x + 1, y + 1),
                block_chi_ratio=self.block_chi_ratio,
                ncv_ratio=self.ncv_ratio,
                rtol=self.cutoff,
                degen_ratio=self.degen_ratio,
            )
            self._env.store_left_projectors(x, y + 1, P, Pt)

        # 2) renormalize every non-equivalent C4, T4 and C1
        for x, y in self._site_coords:
            P = self._env.get_left_P(x, y - 1)
            Pt = self._env.get_left_Pt(x, y)
            nC4 = renormalize_C4_left(
                self._env.get_C4(x, y), self._env.get_T3(x + 1, y), P
            )

            A = self._env.get_A(x + 1, y)
            nT4 = renormalize_T4(Pt, self._env.get_T4(x, y), A, P)

            nC1 = renormalize_C1_left(
                self._env.get_C1(x, y), self._env.get_T1(x + 1, y), Pt
            )
            self._env.store_renormalized_tensors(x + 1, y, nC4, nT4, nC1)

        # 3) store renormalized tensors in the environment
        self._env.set_renormalized_tensors_left()
