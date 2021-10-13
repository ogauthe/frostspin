import numpy as np

from ctmrg import rdm, observables
from ctmrg.ctm_environment import CTM_Environment
from ctmrg.ctm_contract import (
    contract_ul_corner_monolayer,
    contract_ur_corner_monolayer,
    contract_dl_corner_monolayer,
    contract_dr_corner_monolayer,
)
from ctmrg.ctm_renormalize import (
    construct_projectors,
    renormalize_C1_up,
    renormalize_C2_up,
    renormalize_C2_right,
    renormalize_C3_right,
    renormalize_C3_down,
    renormalize_C4_down,
    renormalize_C4_left,
    renormalize_C1_left,
    renormalize_T1_monolayer,
    renormalize_T2_monolayer,
    renormalize_T3_monolayer,
    renormalize_T4_monolayer,
)


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

    def __init__(self, env, chi_setpoint, cutoff, degen_ratio, window, verbosity):
        """
        Constructor for totally asymmetric CTMRG algorithm. Consider using from_file or
        from_elementary_tensors methods instead of calling this one directly.

        Parameters
        ----------
        env: CTM_Environment
            Environment object, as construced by from_file or from_elementary_tensors.
        chi_setpoint : integer
            Maximal corner dimension. This is a setpoint, actual corner dimension
            may be smaller due to cutoff or slightly larger to keep multiplets.
        cutoff : float
            Singular value cutoff to improve stability.
        degen_ratio : float
            Used to define multiplets in projector singular values and truncate between
            two multiplets. Two consecutive (decreasing) values are considered
            degenerate if 1 >= s[i+1]/s[i] >= degen_ratio > 0.
        window : int
            In projector construction, compute chi_setpoint + window singular values
            to preserve multiplet structure.
        verbosity : int
            Level of log verbosity.
        """
        self.verbosity = verbosity
        print(" *** DEV MODE ***")
        if self.verbosity > 0:
            print(f"initalize CTMRG with verbosity = {self.verbosity}")
        self._env = env
        self.chi_setpoint = chi_setpoint
        self.cutoff = cutoff
        self.window = window
        self.degen_ratio = degen_ratio
        self._neq_coords = self._env.neq_coords
        if self.verbosity > 0:
            print(self)
            if self.verbosity > 2:
                self.print_tensor_shapes()

    @classmethod
    def from_elementary_tensors(
        cls,
        tensors,
        tiling,
        chi_setpoint,
        cutoff=0.0,
        degen_ratio=1.0,
        window=0,
        verbosity=0,
    ):
        """
        Construct CTMRG from elementary tensors and tiling.

        Parameters
        ----------
        tensors : enumerable of tensors
            Elementary tensors of unit cell, from left to right from top to bottom.
        tiling : string
            String defining the shape of the unit cell, typically "A" or "AB\nCD".

        Refer to __init__ for the other parameters.
        """
        if verbosity > 0:
            print("Start CTMRG from scratch using elementary tensors")
        env = CTM_Environment.from_elementary_tensors(tensors, tiling)
        return cls(env, chi_setpoint, cutoff, degen_ratio, window, verbosity)

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
        with np.load(filename) as fin:
            try:
                chi_setpoint = fin["_CTM_chi_setpoint"][()]
            except KeyError:  # old data format
                chi_setpoint = fin["_CTM_chi"][()]
            try:
                cutoff = fin["_CTM_cutoff"][()]
                degen_ratio = fin["_CTM_degen_ratio"][()]
                window = fin["_CTM_window"][()]
            except KeyError:  # old data format
                cutoff = 0.0
                degen_ratio = 1.0
                window = 0
        # env construction can take a lot of time (A-A* contraction is expensive)
        # better to open and close savefile twice (here and in env) to have env __init__
        # outside of file opening block.
        env = CTM_Environment.from_file(filename)
        return cls(env, chi_setpoint, cutoff, degen_ratio, window, verbosity)

    def save_to_file(self, filename, additional_data={}):
        """
        Save CTMRG data in external file.

        Parameters
        ----------
        filename: str
            Path to file.
        additional_data: dict
            Data to store together with environment data. Keys have to be string type.
        """
        data = {
            "_CTM_chi_setpoint": self.chi_setpoint,
            "_CTM_cutoff": self.cutoff,
            "_CTM_degen_ratio": self.degen_ratio,
            "_CTM_window": self.window,
        }
        env_data = self._env.get_data_to_save()
        np.savez_compressed(filename, **data, **env_data, **additional_data)
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

    @property
    def Dmax(self):
        return self._env.Dmax

    @property
    def chi_max(self):  # maximal corner dimension, may differ from chi_setpoint
        return self._env.chi_max

    def __repr__(self):
        return f"asymmetric CTMRG with Dmax = {self.Dmax} and chi_max = {self.chi_max}"

    def __str__(self):
        return "\n".join(
            (
                repr(self),
                f"chi_setpoint = {self.chi_setpoint}, cutoff = {self.cutoff}",
                f"degen_ratio = {self.degen_ratio}, window = {self.window}",
                f"unit cell =\n{self._env.cell}",
            )
        )

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
        if self.verbosity > 0:
            print(self)

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

    def compute_rdm_diag_dr(self, x=0, y=0):
        if self.verbosity > 1:
            print(
                f"Compute rdm for down right diagonal sites ({x+1},{y+1}) and",
                f"({x+2},{y+2})",
            )
        return rdm.rdm_diag_dr(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self.construct_reduced_ur(x, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self.construct_reduced_dl(x, y),
            self._env.get_A(x + 2, y + 2),
            self._env.get_T2(x + 3, y + 2),
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
            self.construct_reduced_ul(x, y),
            self._env.get_T1(x + 2, y),
            self._env.get_C2(x + 3, y),
            self._env.get_A(x + 2, y + 1),
            self._env.get_T2(x + 3, y + 1),
            self._env.get_T4(x, y + 2),
            self._env.get_A(x + 1, y + 2),
            self.construct_reduced_dr(x, y),
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
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

    def compute_transfer_spectrum_h(
        self, nval, y=0, v0=None, ncv=None, maxiter=1000, tol=0
    ):
        """
        Compute horizontal transfer matrix spectrum for row y.
        """
        T1s = []
        T3s = []
        for x in range(self.Lx):
            T1s.append(self._env.get_T1(x, y).toarray())
            T3s.append(self._env.get_T3(x, y + 1).toarray())
        return observables.compute_mps_transfer_spectrum(
            T1s, T3s, nval, v0=v0, ncv=ncv, maxiter=maxiter, tol=tol
        )

    def compute_transfer_spectrum_v(
        self, nval, x=0, v0=None, ncv=None, maxiter=1000, tol=0
    ):
        """
        Compute vertical transfer matrix spectrum for column x.
        """
        T2s = []
        T4s = []
        for y in range(self.Ly):
            T2s.append(self._env.get_T2(x + 1, y).toarray().transpose(1, 2, 3, 0))
            T4s.append(self._env.get_T4(x, y).toarray().transpose(1, 2, 3, 0))
        return observables.compute_mps_transfer_spectrum(
            T2s, T4s, nval, v0=v0, ncv=ncv, maxiter=maxiter, tol=tol
        )

    def compute_corr_length_h(self, y=0, v0=None, ncv=None, maxiter=1000, tol=0):
        """
        Compute maximal horizontal correlation length in row between y and y+1.
        """
        _, v2 = self.compute_transfer_spectrum_h(
            2, y, v0=v0, ncv=ncv, maxiter=maxiter, tol=tol
        )
        xi = -self.Lx / np.log(np.abs(v2))
        return xi

    def compute_corr_length_v(self, x=0, v0=None, ncv=None, maxiter=1000, tol=0):
        """
        Compute maximal vertical correlation length in column between x and x+1.
        """
        _, v2 = self.compute_transfer_spectrum_v(
            2, x, v0=v0, ncv=ncv, maxiter=maxiter, tol=tol
        )
        xi = -self.Ly / np.log(np.abs(v2))
        return xi


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

    @classmethod  # no specialized __init__ required
    def from_elementary_tensors(
        cls,
        tiling,
        tensors,
        representations,
        chi_setpoint,
        cutoff=0.0,
        degen_ratio=1.0,
        window=0,
        load_env=None,
        verbosity=0,
    ):
        """
        Construct U(1) symmetric CTMRG from elementary tensors and tiling. Symmetry is
        NOT checked for elementary tensors.

        Parameters
        ----------
        tensors : enumerable of tensors
            Elementary tensors of unit cell, from left to right from top to bottom.
        representations: enumerable of tuple of representation matching tensors.
            Representation for each tensors axis.
        tiling : string
            String defining the shape of the unit cell, typically "A" or "AB\nCD".
        chi_setpoint : integer
            Maximal corner dimension. This is a setpoint, actual corner dimension
            may be smaller due to cutoff or slightly larger to keep multiplets.
        cutoff : float
            Singular value cutoff to improve stability. Default is 0.0 (no cutoff)
        degen_ratio : float
            Used to define multiplets in projector singular values and truncate between
            two multiplets. Two consecutive (decreasing) values are considered
            degenerate if 1 >= s[i+1]/s[i] >= degen_ratio > 0. Default is 1.0 (exact
            degeneracies)
        window : int
            During projector construction, compute chi_setpoint + window singular values
            in each irrep block to preserve global multiplet structure. Required if
            implemented symmetry is smaller than physial symmetry. Default is 0. Can be
            kept to 0 if no degeneracies exist within a given irrep block (e.g. U(1) as
            SU(2) subgroup).
        load_env : string
            File to restart corner and edge environment tensors from, independently
            from elementary tensors. If None, environment tensors will be initalized
            from elementary tensors.
        verbosity : int
            Level of log verbosity. Default is no log.
        """
        if verbosity > 0:
            print("Start CTMRG from elementary tensors")
            if load_env is None:
                print("Initialize environment tensors from elementary tensors")
            else:
                print(f"Load environment from file {load_env}")
        env = CTM_Environment.from_elementary_tensors(
            tiling, tensors, representations, load_env=load_env
        )
        return cls(env, chi_setpoint, cutoff, degen_ratio, window, verbosity)

    def __repr__(self):
        return (
            f"U(1) symmetric CTMRG with Dmax = {self.Dmax} and chi_max = "
            f"{self.chi_max}"
        )

    def set_tensors(self, tensors, representations):
        """
        Set new elementary tensors while keeping current environment if possible.
        """
        if self.verbosity > 0:
            print("set new tensors")
        self._env.set_tensors(tensors, representations)
        if self.verbosity > 0:
            print(self)

    def construct_reduced_dr(self, x, y):
        """
        Return down right corner reduced to U(1) blocks as a U1_SymmetricTensor. Check
        _env to find an already computed corner, if it does not exist construct it and
        store it in _env.

        Unusual leg ordering: reduced corners are constructed to be contracted as
        ul-01-ur
        |      |
        1      0
        0      1
        |      |
        dl-10-dr
        meaning reduced_dr is transposed when compared to standard clockwise order.
        """
        dr = self._env.get_corner_dr(x, y)
        if dr is not None:
            return dr
        dr = contract_dr_corner_monolayer(
            self._env.get_A(x + 2, y + 2),
            self._env.get_T2(x + 3, y + 2),
            self._env.get_T3(x + 2, y + 3),
            self._env.get_C3(x + 3, y + 3),
        )
        self._env.set_corner_dr(x, y, dr)
        return dr

    def construct_reduced_dl(self, x, y):
        """
        Return down left corner reduced to U(1) blocks as a U1_SymmetricTensor. Check
        _env to find an already computed corner, if it does not exist construct it and
        store it in _env.
        """
        dl = self._env.get_corner_dl(x, y)
        if dl is not None:
            return dl
        dl = contract_dl_corner_monolayer(
            self._env.get_T4(x, y + 2),
            self._env.get_A(x + 1, y + 2),
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
        )
        self._env.set_corner_dl(x, y, dl)
        return dl

    def construct_reduced_ul(self, x, y):
        """
        Return upper left corner reduced to U(1) blocks as a U1_SymmetricTensor. Check
        _env to find an already computed corner, if it does not exist construct it and
        store it in _env.
        """
        ul = self._env.get_corner_ul(x, y)
        if ul is not None:
            return ul
        ul = contract_ul_corner_monolayer(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
        )
        self._env.set_corner_ul(x, y, ul)
        return ul

    def construct_reduced_ur(self, x, y):
        """
        Return upper right corner reduced to U(1) blocks as a U1_SymmetricTensor. Check
        _env to find an already computed corner, if it does not exist construct it and
        store it in _env.
        """
        ur = self._env.get_corner_ur(x, y)
        if ur is not None:
            return ur
        ur = contract_ur_corner_monolayer(
            self._env.get_T1(x + 2, y),
            self._env.get_C2(x + 3, y),
            self._env.get_A(x + 2, y + 1),
            self._env.get_T2(x + 3, y + 1),
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
            P, Pt = construct_projectors(
                reduced_dr,
                reduced_ur,
                reduced_ul,
                reduced_dl,
                self.chi_setpoint,
                self.cutoff,
                self.degen_ratio,
                self.window,
            )
            self._env.store_projectors(x + 2, y, P, Pt)

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

            A = self._env.get_A(x, y + 1)
            nT1 = renormalize_T1_monolayer(Pt, self._env.get_T1(x, y), A, P)

            nC2 = renormalize_C2_up(
                self._env.get_C2(x, y), self._env.get_T2(x, y + 1), Pt
            )
            self._env.store_renormalized_tensors(x, y + 1, nC1, nT1, nC2)

        # 3) store renormalized tensors in the environment
        # renormalization reads C1[x,y] but writes C1[x,y+1]
        # => need to compute every renormalized tensors before storing any of them
        self._env.set_renormalized_tensors_up()  # also removes corners ul and ur
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
            P, Pt = construct_projectors(
                reduced_dl,
                reduced_dr,
                reduced_ur,
                reduced_ul,
                self.chi_setpoint,
                self.cutoff,
                self.degen_ratio,
                self.window,
            )
            self._env.store_projectors(x + 3, y + 2, P, Pt)

        # 2) renormalize tensors by absorbing column
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y + 1)
            Pt = self._env.get_Pt(x, y)
            nC2 = renormalize_C2_right(
                self._env.get_C2(x, y), self._env.get_T1(x - 1, y), P
            )

            A = self._env.get_A(x - 1, y)
            nT2 = renormalize_T2_monolayer(Pt, self._env.get_T2(x, y), A, P)

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
        if self.verbosity > 1:
            print("\nstart down move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._neq_coords:
            reduced_ul = self.construct_reduced_ul(x, y)
            reduced_dl = self.construct_reduced_dl(x, y)
            reduced_dr = self.construct_reduced_dr(x, y)
            reduced_ur = self.construct_reduced_ur(x, y)
            P, Pt = construct_projectors(
                reduced_ul,
                reduced_dl,
                reduced_dr,
                reduced_ur,
                self.chi_setpoint,
                self.cutoff,
                self.degen_ratio,
                self.window,
            )
            self._env.store_projectors(x + 3, y + 3, P, Pt)

        # 2) renormalize every non-equivalent C3, T3 and C4
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x - 1, y)
            Pt = self._env.get_Pt(x, y)
            nC3 = renormalize_C3_down(
                self._env.get_C3(x, y), self._env.get_T2(x, y - 1), P
            )

            A = self._env.get_A(x, y - 1)
            nT3 = renormalize_T3_monolayer(Pt, self._env.get_T3(x, y), A, P)

            nC4 = renormalize_C4_down(
                self._env.get_C4(x, y), self._env.get_T4(x, y - 1), Pt
            )
            self._env.store_renormalized_tensors(x, y - 1, nC3, nT3, nC4)

        # 3) store renormalized tensors in the environment
        self._env.set_renormalized_tensors_down()
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
            P, Pt = construct_projectors(
                reduced_ur,
                reduced_ul,
                reduced_dl,
                reduced_dr,
                self.chi_setpoint,
                self.cutoff,
                self.degen_ratio,
                self.window,
            )

            self._env.store_projectors(x, y + 1, P, Pt)

        # 2) renormalize every non-equivalent C4, T4 and C1
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._neq_coords:
            P = self._env.get_P(x, y - 1)
            Pt = self._env.get_Pt(x, y)
            nC4 = renormalize_C4_left(
                self._env.get_C4(x, y), self._env.get_T3(x + 1, y), P
            )

            A = self._env.get_A(x + 1, y)
            nT4 = renormalize_T4_monolayer(Pt, self._env.get_T4(x, y), A, P)

            nC1 = renormalize_C1_left(
                self._env.get_C1(x, y), self._env.get_T1(x + 1, y), Pt
            )
            self._env.store_renormalized_tensors(x + 1, y, nC4, nT4, nC1)

        # 3) store renormalized tensors in the environment
        self._env.set_renormalized_tensors_left()
        if self.verbosity > 1:
            print("left move completed")
