import rdm
import observables
from ctm_environment import CTM_Environment
from ctm_contract import (
    contract_ul_corner_U1,
    contract_ur_corner_U1,
    contract_dl_corner_U1,
    contract_dr_corner_U1,
    contract_u_half,
    contract_l_half,
    contract_d_half,
    contract_r_half,
)
from ctm_renormalize import (
    construct_projectors,
    construct_projectors_U1,
    renormalize_C1_up,
    renormalize_T1,
    renormalize_T1_U1,
    renormalize_C2_up,
    renormalize_C2_right,
    renormalize_T2,
    renormalize_T2_U1,
    renormalize_C3_right,
    renormalize_C3_down,
    renormalize_T3,
    renormalize_T3_U1,
    renormalize_C4_down,
    renormalize_C4_left,
    renormalize_T4,
    renormalize_T4_U1,
    renormalize_C1_left,
)
from toolsU1 import combine_colors, checkU1


class CTMRG(object):
    """
    Corner Transfer Matrix Renormalization Group algorithm. Approximately contract
    a square tensor network with precision controlled by corner dimension chi. This
    implementation is designed for quantum systems where the elementary tensor has a
    bi-layer structure. The tensor can describe both a thermal ensemble, with an
    additional ancilla leg, or a pure wavefunction. It accepts arbitrary unit cells, no
    assumption is made on the dimensions of inequivalent legs.

    Conventions:
    - when passed as arguments to a function for contraction, tensors are sorted from
      left to right, from top to bottom.
    - leg ordering is always (physical, ancilla, *virtual)
    - d stands for the physical dimension
    - a stands for the ancilla dimension (a=d for a thermal ensemble and a=1 for a pure
      wavefunction)
    - Dx stand for the virtual bond dimensions.
    - virtual legs and tensor names are taken clockwise starting from upper right:

        C1-T1-T1-C2
        |  |  |   |
        T4-a--a--T2
        |  |  |   |
        T4-a--a--T2
        |  |  |   |
        C4-T3-T3-C3

    The environment tensors are stored in a custom CTM_Environment class to easily deal
    with periodic boundary conditions and non-equivalent coordinates. Tensor
    contractions, renormalization and reduced density matrix construction functions are
    defined in distinct modules. This class has very few members and makes no
    computation, it is mostly interface and tensor selection.
    """

    def __init__(self, chi, env, verbosity=0):
        """
        Constructor for totally asymmetric CTMRG algorithm. Consider using from_file or
        from_elementary_tensors methods instead of calling this one directly.

        Parameters
        ----------
        chi : integer
          Maximal corner dimension.
        env: CTM_Environment
          Environment object, as construced by from_file or from_elementary_tensors.
        verbosity : int
          Level of log verbosity. Default is no log.
        """
        self.verbosity = verbosity
        if self.verbosity > 0:
            print(f"initalize CTMRG with chi = {chi} and verbosity = {self.verbosity}")
        self.chi = chi
        self._env = env
        self._neq_coords = self._env.neq_coords
        if self.verbosity > 0:
            print("CTMRG constructed")
            print("unit cell =", self._env.cell, sep="\n")
            self.print_tensor_shapes()

    @classmethod
    def from_elementary_tensors(cls, tensors, tiling, chi, verbosity=0):
        """
        Construct CTMRG from elementary tensors and tiling.

        Parameters
        ----------
        tensors : enumerable of tensors
          Elementary tensors of unit cell, from left to right from top to bottom.
        tiling : string
          String defining the shape of the unit cell, typically "A" or "AB\nCD".
        chi : integer
          Maximal corner dimension.
        verbosity : int
          Level of log verbosity. Default is no log.
        """
        if verbosity > 0:
            print("Start CTMRG from scratch using elementary tensors")
        env = CTM_Environment.from_elementary_tensors(tensors, tiling)
        return cls(chi, env, verbosity)

    @classmethod
    def from_file(cls, filename, verbosity=0):
        """
        Construct CTMRG from npz save file, as produced by save_to_file.
        Parameters
        ----------
        filename : str
          Path to npz file
        verbosity : int
          Level of log verbosity. Default is no log.
       """
        if verbosity > 0:
            print("Restart CTMRG from file", filename)
        chi, env = CTM_Environment.from_file(filename)
        return cls(chi, env, verbosity)

    def save_to_file(self, filename, additional_data={}):
        """
        Save CTMRG data in file. If file is not provided, a data dictionary is returned.
        """
        self._env.save_to_file(filename, self.chi, additional_data)
        if self.verbosity > 0:
            print("CTMRG saved in file", filename)

    @property
    def Lx(self):
        return self._env.Lx

    @property
    def Ly(self):
        return self._env.Ly

    @property
    def cell(self):
        return self._env.cell

    @property
    def neq_coords(self):
        return self._neq_coords

    @property
    def cell_number_neq_sites(self):
        return self._env.Nneq

    def restart_environment(self):
        """
        Restart environment tensors from elementary ones. ERASE current environment,
        use with caution.
        """
        if self.verbosity > 0:
            print("Restart brand new environment from elementary tensors.")
        self._env.restart()

    def set_tensors(self, tensors, keep_env=True):
        if keep_env:
            if self.verbosity > 0:
                print("set new tensors")
            self._env.set_tensors(tensors)
        else:  # restart from scratch
            if self.verbosity > 0:
                print("Restart with new tensors and new environment")
            tiling = "\n".join("".join(s) for s in self.cell)
            self._env = CTM_Environment.from_elementary_tensors(tensors, tiling)

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

    def iterate(self):
        if self.verbosity > 1:
            print("Begin CTM iteration")
        self.up_move()
        if self.verbosity > 2:
            self.print_tensor_shapes()
        self.right_move()
        if self.verbosity > 2:
            self.print_tensor_shapes()
        self.down_move()
        if self.verbosity > 2:
            self.print_tensor_shapes()
        self.left_move()
        if self.verbosity > 2:
            self.print_tensor_shapes()
        if self.verbosity > 1:
            print("Finished CTM iteration")

    def converge(self, tol, warmup=0, maxiter=100):
        """
        Converge CTMRG with criterion Frobenius norm(rho) < tol, where rho is the
        average of all first neighbor density matrices in the unit cell. Avoid expensive
        computation of larger density matrices as well as selecting one observable.
        """
        if self.verbosity > 0:
            print(
                f"Converge CTMRG with tol = {tol}, warmup = {warmup},",
                f"maxiter = {maxiter}",
            )
        for i in range(warmup):
            self.iterate()
        if self.verbosity > 0:
            print(f"{warmup} warmup iterations done")
        (rdm2x1_cell, rdm1x2_cell) = self.compute_rdm_1st_neighbor_cell()
        last_rho = (sum(rdm2x1_cell) + sum(rdm1x2_cell)) / self._env.Nneq

        last_last_rho = last_rho
        for i in range(warmup + 1, maxiter + 1):
            self.iterate()
            (rdm2x1_cell, rdm1x2_cell) = self.compute_rdm_1st_neighbor_cell()
            rho = (sum(rdm2x1_cell) + sum(rdm1x2_cell)) / self._env.Nneq
            r = ((last_rho - rho) ** 2).sum() ** 0.5  # shape never changes: 2 <=> inf
            ret = (i, rdm2x1_cell, rdm1x2_cell)
            if self.verbosity > 0:
                print(f"i = {i}, ||rho - last_rho|| = {r}")
            if r < tol:
                return ret  # avoid computing rdm 1st neighbor twice
            if ((last_last_rho - rho) ** 2).sum() ** 0.5 < tol / 100:
                raise RuntimeError("CTMRG oscillates between two converged states", ret)
            last_last_rho = last_rho
            last_rho = rho
        raise RuntimeError("CTMRG did not converge in maxiter", ret)

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
            P, Pt = construct_projectors(R, Rt, self.chi)
            self._env.store_projectors(
                x + 2, y, P, Pt
            )  # indices: Pt <=> renormalized T in R
            del R, Rt

        # 2) renormalize every non-equivalent C1, T1 and C2
        # need all projectors to be constructed at this time
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x + 1, y)
            Pt = self._env.get_Pt(x, y)
            nC1 = renormalize_C1_up(
                self._env.get_C1(x, y), self._env.get_T4(x, y + 1), P
            )

            nT1 = renormalize_T1(
                Pt, self._env.get_T1(x, y), self._env.get_A(x, y + 1), P
            )

            nC2 = renormalize_C2_up(
                self._env.get_C2(x, y), self._env.get_T2(x, y + 1), Pt
            )
            self._env.store_renormalized_tensors(x, y + 1, nC1, nT1, nC2)

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
            P, Pt = construct_projectors(R, Rt, self.chi)
            self._env.store_projectors(x + 3, y + 2, P, Pt)
            del R, Rt

        # 2) renormalize tensors by absorbing column
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y + 1)
            Pt = self._env.get_Pt(x, y)
            nC2 = renormalize_C2_right(
                self._env.get_C2(x, y), self._env.get_T1(x - 1, y), P
            )

            nT2 = renormalize_T2(
                Pt, self._env.get_A(x - 1, y), self._env.get_T2(x, y), P
            )

            nC3 = renormalize_C3_right(
                self._env.get_C3(x, y), self._env.get_T3(x - 1, y), Pt
            )
            self._env.store_renormalized_tensors(x - 1, y, nC2, nT2, nC3)

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

            P, Pt = construct_projectors(R, Rt, self.chi)
            self._env.store_projectors(x + 3, y + 3, P, Pt)
            del R, Rt

        # 2) renormalize every non-equivalent C3, T3 and C4
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x - 1, y)
            Pt = self._env.get_Pt(x, y)
            nC3 = renormalize_C3_down(
                self._env.get_C3(x, y), self._env.get_T2(x, y - 1), P
            )

            nT3 = renormalize_T3(
                Pt, self._env.get_T3(x, y), self._env.get_A(x, y - 1), P
            )

            nC4 = renormalize_C4_down(
                self._env.get_C4(x, y), self._env.get_T4(x, y - 1), Pt
            )
            self._env.store_renormalized_tensors(x, y - 1, nC3, nT3, nC4)

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
            P, Pt = construct_projectors(R, Rt, self.chi)
            self._env.store_projectors(x, y + 1, P, Pt)
            del R, Rt

        # 2) renormalize every non-equivalent C4, T4 and C1
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y - 1)
            Pt = self._env.get_Pt(x, y)
            nC4 = renormalize_C4_left(
                self._env.get_C4(x, y), self._env.get_T3(x + 1, y), P
            )

            nT4 = renormalize_T4(
                Pt, self._env.get_T4(x, y), self._env.get_A(x + 1, y), P
            )

            nC1 = renormalize_C1_left(
                self._env.get_C1(x, y), self._env.get_T1(x + 1, y), Pt
            )
            self._env.store_renormalized_tensors(x + 1, y, nC4, nT4, nC1)

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

    def compute_rdm_1st_neighbor_cell(self):
        """
        Compute reduced density matrix for every couple of inquivalent cell nearest
        neighbor sites.
        """
        if self.verbosity > 1:
            print(
                "Compute reduced density matrix for every cell nearest neighbor sites"
            )
        rdm2x1_cell = []  # avoid import of specific array lib
        rdm1x2_cell = []  # + allow for different d in cell
        for x, y in self._neq_coords:
            rdm2x1_cell.append(self.compute_rdm2x1(x, y))
            rdm1x2_cell.append(self.compute_rdm1x2(x, y))
        return rdm2x1_cell, rdm1x2_cell

    def compute_rdm_2nd_neighbor_cell(self):
        """
        Compute reduced density matrix for every couple of inquivalent cell next nearest
        neighbor sites.
        """
        if self.verbosity > 1:
            print("Compute rdm for every cell next nearest neighbor sites")
        rdm_dr_cell = []
        rdm_ur_cell = []
        for x, y in self._neq_coords:
            rdm_dr_cell.append(self.compute_rdm_diag_dr(x, y))
            rdm_ur_cell.append(self.compute_rdm_diag_ur(x, y))
        return rdm_dr_cell, rdm_ur_cell

    def compute_corr_length_h(self, y=0, v0=None, ncv=None, maxiter=1000, tol=0):
        """
        Compute maximal horizontal correlation length in row between y and y+1.
        """
        T1s = []
        T3s = []
        for x in range(self.Lx):
            T1s.append(self._env.get_T1(x, y))
            T3s.append(self._env.get_T3(x, y + 1))
        return observables.compute_corr_length(T1s, T3s)

    def compute_corr_length_v(self, x=0, v0=None, ncv=None, maxiter=1000, tol=0):
        """
        Compute maximal vertical correlation length in column with between x and x+1.
        """
        T2s = []
        T4s = []
        for y in range(self.Ly):
            T2s.append(self._env.get_T2(x + 1, y).transpose(1, 2, 3, 0))
            T4s.append(self._env.get_T4(x, y).transpose(1, 2, 3, 0))
        return observables.compute_corr_length(T2s, T4s)


class CTMRG_U1(CTMRG):
    """
    Specialized CTMRG algorithm for U(1) symmetric tensors. U(1) symmetry is implemented
    in SVD and cannot be broken during the process. It is also used to speed up certain
    contractions. Refer to CTMRG class for details and conventions.

    Projectors are efficiently constructed using U(1) symmetry. As soon as CTTA corners
    are constructed, they are reduced to a list of submatrices corresponding to U(1)
    blocks. Halves are only constructed blockwise and their SVD is computed on the fly.
    This is efficient since no reshape or transpose happens once corners are
    constructed. However using U(1) in corner construction is not efficient due to
    transposes arising at each step.

    At each move, two corners among ul, ur, dl and dr are renormalized, but the other
    two can still be used for next moves. So every time a corner is computed, store it
    in the environment to be used later (including by compute_rdm_diag ur and dr). After
    each renormalization, delete the two corners that have been renormalized. In total,
    in addition to C and T environment tensors, 4*Lx*Ly corners are stored in reduced
    block form.
    """

    @classmethod  # not specialized __init__ required
    def from_elementary_tensors(cls, tensors, colors, tiling, chi, verbosity=0):
        """
        Construct U(1) symmetric CTMRG from elementary tensors and tiling. Symmetry is
        NOT checked for elementary tensors.

        Parameters
        ----------
        tensors : enumerable of tensors
          Elementary tensors of unit cell, from left to right from top to bottom.
        colors : enumerable of enumerable of integer arrays matching tensors.
          Quantum numbers for elementary tensors.
        tiling : string
          String defining the shape of the unit cell, typically "A" or "AB\nCD".
        chi : integer
          Maximal corner dimension.
        verbosity : int
          Level of log verbosity. Default is no log.
        """
        if verbosity > 0:
            print("Start CTMRG from scratch using elementary tensors")
        env = CTM_Environment.from_elementary_tensors(tensors, tiling, colors)
        return cls(chi, env, verbosity)

    def set_tensors(self, tensors, colors, keep_env=True):
        if keep_env:
            if self.verbosity > 0:
                print("set new tensors")
            self._env.set_tensors(tensors, colors)
        else:  # restart from scratch
            if self.verbosity > 0:
                print("Restart with new tensors and new environment")
            tiling = "\n".join("".join(s) for s in self.cell)
            self._env = CTM_Environment.from_elementary_tensors(tensors, tiling, colors)

    def print_colors(self):
        print("colors_A, colorsC1, colorsC2, colorsC3, colorsC4:")
        for (x, y) in self._neq_coords:
            print(
                f"coords = ({x},{y}), colors are:\nA:",
                *self._env.get_colors_A(x + 1, y + 1),
                "\nC1:",
                self._env.get_color_C1_r(x, y),
                "\n   ",
                self._env.get_color_C1_d(x, y),
                "\nC2:",
                self._env.get_color_C2_l(x + 2, y),
                "\n   ",
                self._env.get_color_C2_d(x + 2, y),
                "\nC3:",
                self._env.get_color_C3_u(x + 2, y + 2),
                "\n   ",
                self._env.get_color_C3_l(x + 2, y + 2),
                "\nC4:",
                self._env.get_color_C4_u(x, y + 2),
                "\n   ",
                self._env.get_color_C4_r(x, y + 2),
            )

    def check_symetries(self):
        """
        Check U(1) symmetry for every unit cell and environment tensors and display
        the result.
        """
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

    def construct_reduced_dr(self, x, y):
        """
        Return down right corner reduced to U(1) blocks as a BlockMatrixU1. Check _env
        to find an already computed corner, if it does not exist construct it and store
        it in _env.
        """
        dr = self._env.get_corner_dr(x, y)
        if dr is not None:
            return dr
        col_Adr = self._env.get_colors_A(x + 2, y + 2)
        a_dr, _, col_col_rd = self._env.get_a_col_rd(x + 2, y + 2)
        dr = contract_dr_corner_U1(
            a_dr,
            self._env.get_T2(x + 3, y + 2),
            self._env.get_T3(x + 2, y + 3),
            self._env.get_C3(x + 3, y + 3),
            col_col_rd,
            combine_colors(col_Adr[2], -col_Adr[2]),
            combine_colors(col_Adr[5], -col_Adr[5]),
            self._env.get_color_T2_u(x + 3, y + 2),
            self._env.get_color_T3_l(x + 2, y + 3),
        )
        self._env.set_corner_dr(x, y, dr)
        return dr

    def construct_reduced_dl(self, x, y):
        """
        Return down left corner reduced to U(1) blocks as a BlockMatrixU1. Check _env
        to find an already computed corner, if it does not exist construct it and store
        it in _env.
        """
        dl = self._env.get_corner_dl(x, y)
        if dl is not None:
            return dl
        col_Adl = self._env.get_colors_A(x + 1, y + 2)
        a_dl, _, col_col_dl = self._env.get_a_col_dl(x + 1, y + 2)
        dl = contract_dl_corner_U1(
            self._env.get_T4(x, y + 2),
            a_dl,
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
            self._env.get_color_T4_u(x, y + 2),
            col_col_dl,
            combine_colors(col_Adl[2], -col_Adl[2]),
            combine_colors(col_Adl[3], -col_Adl[3]),
            self._env.get_color_T3_r(x + 1, y + 3),
        )
        self._env.set_corner_dl(x, y, dl)
        return dl

    def construct_reduced_ul(self, x, y):
        """
        Return upper left corner reduced to U(1) blocks as a BlockMatrixU1. Check _env
        to find an already computed corner, if it does not exist construct it and store
        it in _env.
        """
        ul = self._env.get_corner_ul(x, y)
        if ul is not None:
            return ul
        col_Aul = self._env.get_colors_A(x + 1, y + 1)
        a_ul, _, col_col_ul = self._env.get_a_col_ul(x + 1, y + 1)
        ul = contract_ul_corner_U1(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_T4(x, y + 1),
            a_ul,
            self._env.get_color_T1_r(x + 1, y),
            self._env.get_color_T4_d(x, y + 1),
            col_col_ul,
            combine_colors(col_Aul[3], -col_Aul[3]),
            combine_colors(col_Aul[4], -col_Aul[4]),
        )
        self._env.set_corner_ul(x, y, ul)
        return ul

    def construct_reduced_ur(self, x, y):
        """
        Return upper right corner reduced to U(1) blocks as a BlockMatrixU1. Check _env
        to find an already computed corner, if it does not exist construct it and store
        it in _env.
        """
        ur = self._env.get_corner_ur(x, y)
        if ur is not None:
            return ur
        col_Aur = self._env.get_colors_A(x + 2, y + 1)
        a_ur, _, col_col_ur = self._env.get_a_col_ur(x + 2, y + 1)
        ur = contract_ur_corner_U1(
            self._env.get_T2(x + 3, y + 1),
            self._env.get_C2(x + 3, y),
            a_ur,
            self._env.get_T1(x + 2, y),
            self._env.get_color_T2_d(x + 3, y + 1),
            col_col_ur,
            combine_colors(col_Aur[4], -col_Aur[4]),
            combine_colors(col_Aur[5], -col_Aur[5]),
            self._env.get_color_T1_l(x + 2, y),
        )
        self._env.set_corner_ur(x, y, ur)
        return ur

    def up_move(self):
        """
        Absorb a row on the upper side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete ul
        and ur corners which have been renormalized.
        """
        if self.verbosity > 1:
            print("\nstart up move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            reduced_dr = self.construct_reduced_dr(x, y)
            reduced_ur = self.construct_reduced_ur(x, y)
            reduced_ul = self.construct_reduced_ul(x, y)
            reduced_dl = self.construct_reduced_dl(x, y)
            P, Pt, colors = construct_projectors_U1(
                reduced_dr, reduced_ur, reduced_ul, reduced_dl, self.chi
            )
            self._env.store_projectors(x + 2, y, P, Pt, colors)

        # 2) renormalize every non-equivalent C1, T1 and C2
        # need all projectors to be constructed at this time
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x + 1, y)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x + 1, y)
            color_Pt = -self._env.get_color_P(x, y)
            nC1 = renormalize_C1_up(
                self._env.get_C1(x, y), self._env.get_T4(x, y + 1), P
            )

            col_Aul = self._env.get_colors_A(x, y + 1)
            a_ul, _, col_a_ul = self._env.get_a_col_ul(x, y + 1)
            nT1 = renormalize_T1_U1(
                Pt,
                self._env.get_T1(x, y),
                a_ul,
                P,
                self._env.get_color_T1_r(x, y),
                color_Pt,
                col_a_ul,
                combine_colors(col_Aul[3], -col_Aul[3]),
                combine_colors(col_Aul[4], -col_Aul[4]),
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
        self._env.fix_renormalized_up()  # also removes corners ul and ur
        if self.verbosity > 1:
            print("up move completed")

    def right_move(self):
        """
        Absorb a column on the right side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete ur
        and dr corners which have been renormalized.
        """
        if self.verbosity > 1:
            print("\nstart right move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            reduced_dl = self.construct_reduced_dl(x, y)
            reduced_dr = self.construct_reduced_dr(x, y)
            reduced_ur = self.construct_reduced_ur(x, y)
            reduced_ul = self.construct_reduced_ul(x, y)
            P, Pt, colors = construct_projectors_U1(
                reduced_dl, reduced_dr, reduced_ur, reduced_ul, self.chi
            )
            self._env.store_projectors(x + 3, y + 2, P, Pt, colors)

        # 2) renormalize tensors by absorbing column
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y + 1)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x, y + 1)
            color_Pt = -self._env.get_color_P(x, y)
            nC2 = renormalize_C2_right(
                self._env.get_C2(x, y), self._env.get_T1(x - 1, y), P
            )

            col_Aur = self._env.get_colors_A(x - 1, y)
            a_ur, _, col_a_ur = self._env.get_a_col_ur(x - 1, y)
            nT2 = renormalize_T2_U1(
                Pt,
                self._env.get_T2(x, y),
                a_ur,
                P,
                self._env.get_color_T2_d(x, y),
                color_Pt,
                col_a_ur,
                combine_colors(col_Aur[4], -col_Aur[4]),
                combine_colors(col_Aur[5], -col_Aur[5]),
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
        """
        Absorb a row on the down side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete dl
        and dr corners which have been renormalized.
        """
        if self.verbosity > 1:
            print("\nstart down move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            reduced_ul = self.construct_reduced_ul(x, y)
            reduced_dl = self.construct_reduced_dl(x, y)
            reduced_dr = self.construct_reduced_dr(x, y)
            reduced_ur = self.construct_reduced_ur(x, y)
            P, Pt, colors = construct_projectors_U1(
                reduced_ul, reduced_dl, reduced_dr, reduced_ur, self.chi
            )
            self._env.store_projectors(x + 3, y + 3, P, Pt, colors)

        # 2) renormalize every non-equivalent C3, T3 and C4
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x - 1, y)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x - 1, y)
            color_Pt = -self._env.get_color_P(x, y)
            nC3 = renormalize_C3_down(
                self._env.get_C3(x, y), self._env.get_T2(x, y - 1), P
            )

            col_Adl = self._env.get_colors_A(x, y - 1)
            a_dl, _, col_a_dl = self._env.get_a_col_dl(x, y - 1)
            nT3 = renormalize_T3_U1(
                Pt,
                self._env.get_T3(x, y),
                a_dl,
                P,
                self._env.get_color_T3_r(x, y),
                color_P,
                col_a_dl,
                combine_colors(col_Adl[2], -col_Adl[2]),
                combine_colors(col_Adl[3], -col_Adl[3]),
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
        """
        Absorb a column on the left side of environment. Compute new ul / ur / dl / dr
        corners only if they cannot be found in the environment. At the end, delete ul
        and dl corners which have been renormalized.
        """
        if self.verbosity > 1:
            print("\nstart left move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            reduced_ur = self.construct_reduced_ur(x, y)
            reduced_ul = self.construct_reduced_ul(x, y)
            reduced_dl = self.construct_reduced_dl(x, y)
            reduced_dr = self.construct_reduced_dr(x, y)
            P, Pt, colors = construct_projectors_U1(
                reduced_ur, reduced_ul, reduced_dl, reduced_dr, self.chi
            )

            self._env.store_projectors(x, y + 1, P, Pt, colors)

        # 2) renormalize every non-equivalent C4, T4 and C1
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y - 1)
            Pt = self._env.get_Pt(x, y)
            color_P = self._env.get_color_P(x, y - 1)
            color_Pt = -self._env.get_color_P(x, y)
            nC4 = renormalize_C4_left(
                self._env.get_C4(x, y), self._env.get_T3(x + 1, y), P
            )

            col_Aul = self._env.get_colors_A(x + 1, y)
            a_ul, _, col_a_ul = self._env.get_a_col_ul(x + 1, y)
            nT4 = renormalize_T4_U1(
                Pt,
                self._env.get_T4(x, y),
                a_ul,
                P,
                self._env.get_color_T4_d(x, y),
                color_P,
                col_a_ul,
                combine_colors(col_Aul[3], -col_Aul[3]),
                combine_colors(col_Aul[4], -col_Aul[4]),
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
            ur=self.construct_reduced_ur(x, y),
            dl=self.construct_reduced_dl(x, y),
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
            ul=self.construct_reduced_ul(x, y),
            dr=self.construct_reduced_dr(x, y).T,
        )
