import rdm
from ctm_environment import CTM_Environment
from ctm_contract import (
    contract_u_half,
    contract_l_half,
    contract_d_half,
    contract_r_half,
)
from ctm_renormalize import (
    construct_projectors,
    renormalize_C1_up,
    renormalize_T1,
    renormalize_C2_up,
    renormalize_C2_right,
    renormalize_T2,
    renormalize_C3_right,
    renormalize_C3_down,
    renormalize_T3,
    renormalize_C4_down,
    renormalize_C4_left,
    renormalize_T4,
    renormalize_C1_left,
)
from toolsU1 import combine_colors, checkU1


class CTMRG(object):
    """
    Convention: legs and tensors are taken clockwise starting from upper right:

        C1-T1-T1-C2
        |  |  |   |
        T4-a--a--T2
        |  |  |   |
        T4-a--a--T2
        |  |  |   |
        C4-T3-T3-C3

    When passed as arguments to a function for contraction, tensors are sorted from left
    to right, from up to down.
    """

    def __init__(
        self,
        chi,
        tensors=(),
        cell=None,
        tiling=None,
        colors=None,
        file=None,
        verbosity=0,
    ):
        self.verbosity = verbosity
        if self.verbosity > 0:
            print(f"initalize CTMRG with chi = {chi} and verbosity = {self.verbosity}")
        self.chi = chi
        self._env = CTM_Environment(
            tensors, cell=cell, tiling=tiling, colors=colors, file=file
        )
        self._neq_coords = self._env.neq_coords
        if self.verbosity > 0:
            print("CTMRG constructed")
            print("unit cell =", self._env.cell, sep="\n")
            self.print_tensor_shapes()
            if colors is not None:
                print("colors =", colors, sep="\n")

    def save_to_file(self, file=None):
        """
        Save CTMRG data in file. If file is not provided, a data dictionary is returned.
        """
        return self._env.save_to_file(file)  # all the data is in env

    @property
    def Lx(self):
        return self._env.Lx

    @property
    def Ly(self):
        return self._env.Ly

    @property
    def cell(self):
        return self._env.cell

    def set_tensors(self, tensors, colors=None, keep_env=True):
        if self.verbosity > 0:
            print("set new tensors")
        if keep_env:
            self._env.set_tensors(tensors, colors=colors)
        else:  # restart from fresh
            self._env = CTM_Environment(
                tensors, cell=self._env.cell.copy(), colors=colors
            )

    def print_tensor_shapes(self):
        print("tensor shapes for C1 T1 C2 // T4 A T2 // C4 T3 C4:")
        for (x, y) in self._neq_coords:
            print(
                f"({x},{y}):",
                self._env.get_C1(x, y).shape,
                self._env.get_T1(x + 1, y).shape,
                self._env.get_C2(x + 2, y).shape,
                self._env.get_T4(x, y + 1).shape,
                self._env.get_A(x + 1, y + 1).shape,
                self._env.get_T2(x + 2, y + 1).shape,
                self._env.get_C4(x, y + 3).shape,
                self._env.get_T3(x + 1, y + 2).shape,
                self._env.get_C3(x + 2, y + 2).shape,
            )

    def check_symetries(self):
        for (x, y) in self._neq_coords:
            colC1 = (self._env.get_color_C1_r(x, y), self._env.get_color_C1_d(x, y))
            colT1 = (
                self._env.get_color_T1_r(x, y),
                self._env.get_color_T1_d(x, y),
                -self._env.get_color_T1_d(x, y),
                self._env.get_color_T1_l(x, y),
            )
            colC2 = (self._env.get_color_C2_d(x, y), self._env.get_color_C2_l(x, y))
            colT2 = (
                self._env.get_color_T2_u(x, y),
                self._env.get_color_T2_d(x, y),
                self._env.get_color_T2_l(x, y),
                -self._env.get_color_T2_l(x, y),
            )
            colC3 = (self._env.get_color_C3_u(x, y), self._env.get_color_C3_l(x, y))
            colT3 = (
                self._env.get_color_T3_u(x, y),
                -self._env.get_color_T3_u(x, y),
                self._env.get_color_T3_r(x, y),
                self._env.get_color_T3_l(x, y),
            )
            colC4 = (self._env.get_color_C4_u(x, y), self._env.get_color_C4_r(x, y))
            colT4 = (
                self._env.get_color_T4_u(x, y),
                self._env.get_color_T4_r(x, y),
                -self._env.get_color_T4_r(x, y),
                self._env.get_color_T4_d(x, y),
            )
            print(
                f"({x},{y}):",
                "C1",
                checkU1(self._env.get_C1(x, y), colC1),
                "T1",
                checkU1(self._env.get_T1(x, y), colT1),
                "C2",
                checkU1(self._env.get_C2(x, y), colC2),
                "T2",
                checkU1(self._env.get_T2(x, y), colT2),
                "C3",
                checkU1(self._env.get_C3(x, y), colC3),
                "T3",
                checkU1(self._env.get_T3(x, y), colT3),
                "C4",
                checkU1(self._env.get_C4(x, y), colC4),
                "T4",
                checkU1(self._env.get_T4(x, y), colT4),
            )

    def iterate(self):
        self.up_move()
        if self.verbosity > 1:
            self.print_tensor_shapes()
        self.right_move()
        if self.verbosity > 1:
            self.print_tensor_shapes()
        self.down_move()
        if self.verbosity > 1:
            self.print_tensor_shapes()
        self.left_move()
        if self.verbosity > 1:
            self.print_tensor_shapes()

    def converge(self, tol, warmup=0, maxiter=100):
        """
        Converge CTMRG with criterion Frobenius norm(rho) < tol, where rho is the
        average of all first neighbor density matrices in the unit cell. Avoid expensive
        computation of larger density matrices as well as selecting one observable.
        """
        if self.verbosity > 0:
            print(f"Converge CTMRG with tol = {tol}")
        for i in range(warmup):
            self.iterate()
        if self.verbosity > 0:
            print(f"{warmup} warmup iterations done")
        last_rho = self.compute_rdm_cell_average_1st_nei()
        last_last_rho = last_rho
        for i in range(maxiter):
            self.iterate()
            rho = self.compute_rdm_cell_average_1st_nei()
            r = ((last_rho - rho) ** 2).sum() ** 0.5
            if self.verbosity > 0:
                print(f"i = {i}, ||rho - last_rho|| = {r}")
            if r < tol:
                return i + warmup, rho  # avoid computing it twice
            if ((last_last_rho - rho) ** 2).sum() ** 0.5 < tol:
                raise RuntimeError("CTMRG oscillates between two converged states")
            last_last_rho = last_rho
            last_rho = rho
        raise RuntimeError("CTMRG did not converge in maxiter")

    def up_move(self):
        if self.verbosity > 1:
            print("\nstart up move")
        # 1) compute isometries for every non-equivalent sites
        # convention : get projectors from svd(R @ Rt)
        for x, y in self._neq_coords:
            R = contract_r_half(
                self._env.get_T1(x + 2, y),
                self._env.get_C2(x + 3, y),
                self._env.get_A(x + 2, y + 1),
                self._env.get_T2(x + 3, y + 1),
                self._env.get_A(x + 2, y + 2),
                self._env.get_T2(x + 3, y + 2),
                self._env.get_T3(x + 2, y + 3),
                self._env.get_C3(x + 3, y + 3),
            )
            Rt = contract_l_half(
                self._env.get_C1(x, y),
                self._env.get_T1(x + 1, y),
                self._env.get_T4(x, y + 1),
                self._env.get_A(x + 1, y + 1),
                self._env.get_T4(x, y + 2),
                self._env.get_A(x + 1, y + 2),
                self._env.get_C4(x, y + 3),
                self._env.get_T3(x + 1, y + 3),
            )
            #        L-0  == 1-R
            #        L         R
            #        L         R
            #        L-1     0-R
            col_Adr_l = self._env.get_colors_A(x + 2, y + 2)[5]
            ext_colors = combine_colors(
                self._env.get_color_T3_l(x + 2, y + 3), col_Adr_l, -col_Adr_l
            )
            col_Aul_r = self._env.get_colors_A(x + 1, y + 1)[3]
            int_colors = combine_colors(
                self._env.get_color_T1_r(x + 1, y), col_Aul_r, -col_Aul_r
            )
            P, Pt, color = construct_projectors(R, Rt, self.chi, ext_colors, int_colors)
            self._env.store_projectors(
                x + 2, y, P, Pt, color
            )  # indices: Pt <=> renormalized T in R
            del R, Rt

        # 2) renormalize every non-equivalent C1, T1 and C2
        # need all projectors to be constructed at this time
        if self.verbosity > 1:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x + 1, y)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x + 1, y)
            color_Pt = -self._env.get_color_P(x, y)
            nC1 = renormalize_C1_up(
                self._env.get_C1(x, y), self._env.get_T4(x, y + 1), P
            )

            nT1 = renormalize_T1(
                Pt, self._env.get_T1(x, y), self._env.get_A(x, y + 1), P
            )

            nC2 = renormalize_C2_up(
                self._env.get_C2(x, y), self._env.get_T2(x, y + 1), Pt
            )
            self._env.store_renormalized_tensors(
                x, y + 1, nC1, nT1, nC2, color_P, color_Pt
            )

        # 3) store renormalized tensors in the environment
        # renormalization reads C1[x,y] but writes C1[x,y+1]
        # => need to compute every renormalized tensors before storing any of them
        self._env.fix_renormalized_up()
        if self.verbosity > 1:
            print("up move completed")

    def right_move(self):
        if self.verbosity > 1:
            print("\nstart right move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            #      0  1    0
            #      |  | => R
            #      DDDD    1
            R = contract_d_half(
                self._env.get_T4(x, y + 2),
                self._env.get_A(x + 1, y + 2),
                self._env.get_A(x + 2, y + 2),
                self._env.get_T2(x + 3, y + 2),
                self._env.get_C4(x, y + 3),
                self._env.get_T3(x + 1, y + 3),
                self._env.get_T3(x + 2, y + 3),
                self._env.get_C3(x + 3, y + 3),
            )
            #      UUUU     0
            #      |  | =>  Rt
            #      1  0     1
            Rt = contract_u_half(
                self._env.get_C1(x, y),
                self._env.get_T1(x + 1, y),
                self._env.get_T1(x + 2, y),
                self._env.get_C2(x + 3, y),
                self._env.get_T4(x, y + 1),
                self._env.get_A(x + 1, y + 1),
                self._env.get_A(x + 2, y + 1),
                self._env.get_T2(x + 3, y + 1),
            )
            col_Adl_u = self._env.get_colors_A(x + 1, y + 2)[2]
            ext_colors = combine_colors(
                self._env.get_color_T4_u(x, y + 2), col_Adl_u, -col_Adl_u
            )
            col_Aur_d = self._env.get_colors_A(x + 2, y + 1)[4]
            int_colors = combine_colors(
                self._env.get_color_T2_d(x + 3, y + 1), col_Aur_d, -col_Aur_d
            )
            P, Pt, color = construct_projectors(R, Rt, self.chi, ext_colors, int_colors)
            self._env.store_projectors(x + 3, y + 2, P, Pt, color)
            del R, Rt

        # 2) renormalize tensors by absorbing column
        if self.verbosity > 1:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y + 1)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x, y + 1)
            color_Pt = -self._env.get_color_P(x, y)
            nC2 = renormalize_C2_right(
                self._env.get_C2(x, y), self._env.get_T1(x - 1, y), P
            )

            nT2 = renormalize_T2(
                Pt, self._env.get_A(x - 1, y), self._env.get_T2(x, y), P
            )

            nC3 = renormalize_C3_right(
                self._env.get_C3(x, y), self._env.get_T3(x - 1, y), Pt
            )
            self._env.store_renormalized_tensors(
                x - 1, y, nC2, nT2, nC3, color_P, color_Pt
            )

        # 3) store renormalized tensors in the environment
        self._env.fix_renormalized_right()
        if self.verbosity > 1:
            print("right move completed")

    def down_move(self):
        if self.verbosity > 1:
            print("\nstart down move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            #        L-0      L-1
            #        L        L
            #        L    =>  L
            #        L-1      L-0
            R = contract_l_half(
                self._env.get_C1(x, y),
                self._env.get_T1(x + 1, y),
                self._env.get_T4(x, y + 1),
                self._env.get_A(x + 1, y + 1),
                self._env.get_T4(x, y + 2),
                self._env.get_A(x + 1, y + 2),
                self._env.get_C4(x, y + 3),
                self._env.get_T3(x + 1, y + 3),
            )
            #      1-R
            #        R
            #        R
            #      0-R
            Rt = contract_r_half(
                self._env.get_T1(x + 2, y),
                self._env.get_C2(x + 3, y),
                self._env.get_A(x + 2, y + 1),
                self._env.get_T2(x + 3, y + 1),
                self._env.get_A(x + 2, y + 2),
                self._env.get_T2(x + 3, y + 2),
                self._env.get_T3(x + 2, y + 3),
                self._env.get_C3(x + 3, y + 3),
            )

            col_Aul_r = self._env.get_colors_A(x + 1, y + 1)[3]
            ext_colors = combine_colors(
                self._env.get_color_T1_r(x + 1, y), col_Aul_r, -col_Aul_r
            )
            col_Adr_l = self._env.get_colors_A(x + 2, y + 2)[5]
            int_colors = combine_colors(
                self._env.get_color_T3_l(x + 2, y + 3), col_Adr_l, -col_Adr_l
            )
            P, Pt, color = construct_projectors(R, Rt, self.chi, ext_colors, int_colors)
            self._env.store_projectors(x + 3, y + 3, P, Pt, color)
            del R, Rt

        # 2) renormalize every non-equivalent C3, T3 and C4
        if self.verbosity > 1:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x - 1, y)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x - 1, y)
            color_Pt = -self._env.get_color_P(x, y)
            nC3 = renormalize_C3_down(
                self._env.get_C3(x, y), self._env.get_T2(x, y - 1), P
            )

            nT3 = renormalize_T3(
                Pt, self._env.get_T3(x, y), self._env.get_A(x, y - 1), P
            )

            nC4 = renormalize_C4_down(
                self._env.get_C4(x, y), self._env.get_T4(x, y - 1), Pt
            )
            self._env.store_renormalized_tensors(
                x, y - 1, nC3, nT3, nC4, color_P, color_Pt
            )

        # 3) store renormalized tensors in the environment
        self._env.fix_renormalized_down()
        if self.verbosity > 1:
            print("down move completed")

    def left_move(self):
        if self.verbosity > 1:
            print("\nstart left move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            #      UUUU      1
            #      |  |  =>  R
            #      1  0      0
            R = contract_u_half(
                self._env.get_C1(x, y),
                self._env.get_T1(x + 1, y),
                self._env.get_T1(x + 2, y),
                self._env.get_C2(x + 3, y),
                self._env.get_T4(x, y + 1),
                self._env.get_A(x + 1, y + 1),
                self._env.get_A(x + 2, y + 1),
                self._env.get_T2(x + 3, y + 1),
            )
            #      0  1
            #      |  |
            #      DDDD
            Rt = contract_d_half(
                self._env.get_T4(x, y + 2),
                self._env.get_A(x + 1, y + 2),
                self._env.get_A(x + 2, y + 2),
                self._env.get_T2(x + 3, y + 2),
                self._env.get_C4(x, y + 3),
                self._env.get_T3(x + 1, y + 3),
                self._env.get_T3(x + 2, y + 3),
                self._env.get_C3(x + 3, y + 3),
            )
            col_Aur_d = self._env.get_colors_A(x + 2, y + 1)[4]
            ext_colors = combine_colors(
                self._env.get_color_T2_d(x + 3, y + 1), col_Aur_d, -col_Aur_d
            )
            col_Adl_u = self._env.get_colors_A(x + 1, y + 2)[2]
            int_colors = combine_colors(
                self._env.get_color_T4_u(x, y + 2), col_Adl_u, -col_Adl_u
            )
            P, Pt, color = construct_projectors(R, Rt, self.chi, ext_colors, int_colors)
            self._env.store_projectors(x, y + 1, P, Pt, color)
            del R, Rt

        # 2) renormalize every non-equivalent C4, T4 and C1
        if self.verbosity > 1:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y - 1)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x, y - 1)
            color_Pt = -self._env.get_color_P(x, y)
            nC4 = renormalize_C4_left(
                self._env.get_C4(x, y), self._env.get_T3(x + 1, y), P
            )

            nT4 = renormalize_T4(
                Pt, self._env.get_T4(x, y), self._env.get_A(x + 1, y), P
            )

            nC1 = renormalize_C1_left(
                self._env.get_C1(x, y), self._env.get_T1(x + 1, y), Pt
            )
            self._env.store_renormalized_tensors(
                x + 1, y, nC4, nT4, nC1, color_P, color_Pt
            )

        # 3) store renormalized tensors in the environment
        self._env.fix_renormalized_left()
        if self.verbosity > 1:
            print("left move completed")

    def compute_rdm1x1(self, x=0, y=0):
        if self.verbosity > 1:
            print(f"Compute rdm 1x1 with C1 coord = ({x},{y})")
        return rdm.rdm_1x1(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_C2(x + 2, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_T2(x + 2, y + 1),
            self._env.get_C4(x, y + 2),
            self._env.get_T3(x + 1, y + 2),
            self._env.get_C3(x + 2, y + 2),
        )

    def compute_rdm1x2(self, x=0, y=0):
        if self.verbosity > 1:
            print(f"Compute rdm 1x2 with C1 coord = ({x},{y})")
        return rdm.rdm_1x2(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_T1(x + 2, y),
            self._env.get_C2(x + 3, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_A(x + 2, y + 1),
            self._env.get_T2(x + 3, y + 1),
            self._env.get_C4(x, y + 2),
            self._env.get_T3(x + 1, y + 2),
            self._env.get_T3(x + 2, y + 2),
            self._env.get_C3(x + 3, y + 2),
        )

    def compute_rdm2x1(self, x=0, y=0):
        if self.verbosity > 1:
            print(f"Compute rdm 2x1 with C1 coord = ({x},{y})")
        return rdm.rdm_2x1(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_C2(x + 2, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_T2(x + 2, y + 1),
            self._env.get_T4(x, y + 2),
            self._env.get_A(x + 1, y + 2),
            self._env.get_T2(x + 2, y + 2),
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
            self._env.get_C3(x + 2, y + 3),
        )

    def compute_rdm2x2(self, x=0, y=0):
        if self.verbosity > 1:
            print(f"Compute rdm 2x2 with C1 coord = ({x},{y})")
        return rdm.rdm_2x2(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_T1(x + 2, y),
            self._env.get_C2(x + 3, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_A(x + 2, y + 1),
            self._env.get_T2(x + 3, y + 1),
            self._env.get_T4(x, y + 2),
            self._env.get_A(x + 1, y + 2),
            self._env.get_A(x + 2, y + 2),
            self._env.get_T2(x + 3, y + 2),
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
            self._env.get_T3(x + 2, y + 3),
            self._env.get_C3(x + 3, y + 3),
        )

    def compute_rdm_diag_dr(self, x=0, y=0):
        if self.verbosity > 1:
            print(
                f"Compute rdm for down right diagonal sites ({x+1},{y+1}) and",
                f"({x+2},{y+2})",
            )
        return rdm.rdm_diag_dr(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_T1(x + 2, y),
            self._env.get_C2(x + 3, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_A(x + 2, y + 1),
            self._env.get_T2(x + 3, y + 1),
            self._env.get_T4(x, y + 2),
            self._env.get_A(x + 1, y + 2),
            self._env.get_A(x + 2, y + 2),
            self._env.get_T2(x + 3, y + 2),
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
            self._env.get_T3(x + 2, y + 3),
            self._env.get_C3(x + 3, y + 3),
        )

    def compute_rdm_diag_ur(self, x=0, y=0):
        if self.verbosity > 1:
            print(
                f"Compute rdm for upper right diagonal sites ({x+1},{y+2}) and",
                f"({x+2},{y+1})",
            )
        return rdm.rdm_diag_ur(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_T1(x + 2, y),
            self._env.get_C2(x + 3, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self._env.get_A(x + 2, y + 1),
            self._env.get_T2(x + 3, y + 1),
            self._env.get_T4(x, y + 2),
            self._env.get_A(x + 1, y + 2),
            self._env.get_A(x + 2, y + 2),
            self._env.get_T2(x + 3, y + 2),
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
            self._env.get_T3(x + 2, y + 3),
            self._env.get_C3(x + 3, y + 3),
        )

    def compute_rdm_cell_average_1st_nei(self):
        """
        Compute cell average of first neighbor reduced density matrices.
        """
        if self.verbosity > 0:
            print("Compute rdm average on cell")
        rdm = 0  # no need to initalize shape, avoid import of specific array lib
        for x, y in self._neq_coords:
            rdm = rdm + self.compute_rdm2x1(x, y) + self.compute_rdm1x2(x, y)
        return rdm / len(self._neq_coords)

    def compute_rdm_cell_average_2nd_nei(self):
        """
        Compute cell average of second neighbor reduced density matrices.
        """
        if self.verbosity > 0:
            print("Compute rdm average on cell")
        rdm = 0
        for x, y in self._neq_coords:
            rdm = rdm + self.compute_rdm_diag_dr(x, y) + self.compute_rdm_diag_ur(x, y)
        return rdm / len(self._neq_coords)
