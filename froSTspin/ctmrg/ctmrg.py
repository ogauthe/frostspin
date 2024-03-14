import numpy as np

from . import rdm, observables
from .ctm_environment import CTM_Environment
from .ctm_contract import (
    contract_ul,
    contract_ur,
    contract_dl,
    contract_dr,
)
from .ctm_renormalize import (
    construct_projectors,
    renormalize_C1_up,
    renormalize_C2_up,
    renormalize_C2_right,
    renormalize_C3_right,
    renormalize_C3_down,
    renormalize_C4_down,
    renormalize_C4_left,
    renormalize_C1_left,
    renormalize_T1,
    renormalize_T2,
    renormalize_T3,
    renormalize_T4,
)


class CTMRG:
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

    Virtual symmetries are implemented using SymmetricTensors framework. It is essential
    to speed-up contractions and SVDs, as well as improving numerical precision.

    At each move, two corners among ul, ur, dl and dr are renormalized, but the other
    two can still be used for next moves. Each time a corner is computed, it is stored
    in the environment to be used later (including by compute_rdm_diag ur and dr). After
    each renormalization, delete the two corners that have been renormalized. In total,
    in addition to C and T environment tensors, 4*Lx*Ly corners are stored.
    """

    def __init__(self, env, chi_target, verbosity, **ctm_params):
        """
        Constructor for totally asymmetric CTMRG algorithm. Consider using from_file or
        from_elementary_tensors methods instead of calling this one directly.

        Parameters
        ----------
        env: CTM_Environment
            Environment object, as construced by from_file or from_elementary_tensors.
        chi_target : integer
            Target for corner dimension. This is a target, actual corner dimension chi
            may differ differ independently on a any corner due to cutoff or multiplets.
        verbosity : int
            Level of log verbosity.


        Additional parameters
        ---------------------
        block_chi_ratio: float
            Compute min(chi_target, block_chi_ratio * last_block_chi) singular values
            in each symmetry block during projector construction, where last_block_chi
            is the number of singular values in this block last iteration.
            Default is 1.1.
        ncv_ratio : float
            Compute ncv_ratio * block_chi Lanczos vectors in each symmetry block.
            Default is 2.0.
        cutoff : float
            Singular value cutoff to improve stability.
            Default is 1e-11.
        degen_ratio : float
            Used to define multiplets in projector singular values and truncate between
            two multiplets. Two consecutive (decreasing) values are considered
            degenerate if 1 >= s[i+1]/s[i] >= degen_ratio > 0.
            Default is 0.9999.
        """
        self.verbosity = verbosity
        if self.verbosity > 0:
            print(f"initalize CTMRG with verbosity = {self.verbosity}")
        self._env = env
        self._site_coords = env.site_coords
        self.chi_target = int(chi_target)

        # set parameters.
        self.block_chi_ratio = float(ctm_params.get("block_chi_ratio", 1.1))
        self.ncv_ratio = float(ctm_params.get("block_ncv_ratio", 2.0))
        self.cutoff = float(ctm_params.get("cutoff", 1e-11))
        self.degen_ratio = float(ctm_params.get("degen_ratio", 0.9999))

        if self.verbosity > 0:
            print(self)
            if self.verbosity > 2:
                self.print_tensor_shapes()

        if self.chi_target < 2:
            raise ValueError("chi must be larger than 2")
        if self.block_chi_ratio < 1.0:
            raise ValueError("block_chi_ratio must be larger than 1.0")
        if self.ncv_ratio < 2.0:
            raise ValueError("ncv_ratio must be larger than 2.0")
        if not (0.0 <= self.cutoff < 1.0):
            raise ValueError("cutoff must me obey 0.0 <= self.cutoff < 1.0")
        if not (0.0 < self.degen_ratio <= 1.0):
            raise ValueError("degen_ratio must obey 0.0 <= self.degen_ratio <= 1.0")

        if self.Dmin != self.Dmax:
            print(
                f"WARNING: initialize CTMRG with Dmin = {self.Dmin} != Dmax ="
                f" {self.Dmax}"
            )

    @classmethod
    def from_elementary_tensors(
        cls, tiling, tensors, chi_target, dummy=True, verbosity=0, **ctm_params
    ):
        """
        Construct CTMRG from elementary tensors and tiling.
        If dummy is True (default), environment tensors are initialized as dummy and act
        as identity between bra and ket layers: corners are (1x1) identity matrices with
        trivial representation and edges are identiy matrices between layers with a
        representation matching site tensor, with dummy legs added to match corners.
        If dummy is False, each environment tensor is initalized from double-layer
        elementary site tensor after tracing over directions absent from said tensor.
        This should be equivalent to starting with dummy and running one iteration.

        Parameters
        ----------
        tensors : enumerable of SymmetricTensor
            Elementary tensors of unit cell, from left to right from top to bottom.
        tiling : string
            String defining the shape of the unit cell, typically "A" or "AB\nCD".
        chi_target : integer
            Target for corner dimension. This is a target, actual corner dimension chi
            may differ differ independently on a any corner due to cutoff or multiplets.
        dummy : bool
            Whether to initalize the environment tensors as dummy or from site tensors.
            Default is False.
        verbosity : int
            Level of log verbosity. Default is no log.

        See `__init__` for a description of additional parameters.
        """
        if verbosity > 0:
            if dummy:
                print("Start CTMRG from dummy environment")
            else:
                print("Start CTMRG from elementary tensors")
        env = CTM_Environment.from_elementary_tensors(tiling, tensors, dummy)
        return cls(env, chi_target, verbosity, **ctm_params)

    def set_tensors(self, tensors):
        """
        Set new elementary tensors while keeping current environment if possible.
        """
        if self.verbosity > 0:
            print("set new tensors")
        self._env.set_tensors(tensors)
        if self.verbosity > 2:
            self.print_tensor_shapes()

    @classmethod
    def load_from_file(cls, filename, verbosity=0):
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

        ctm_params = {}
        with np.load(filename) as fin:
            chi_target = int(fin["_CTM_chi_target"])
            ctm_params["block_chi_ratio"] = fin["_CTM_block_chi_ratio"]
            ctm_params["ncv_ratio"] = fin["_CTM_ncv_ratio"]
            ctm_params["cutoff"] = fin["_CTM_cutoff"]
            ctm_params["degen_ratio"] = fin["_CTM_degen_ratio"]
        # better to open and close savefile twice (here and in env) to have env __init__
        # outside of file opening block.
        env = CTM_Environment.load_from_file(filename)
        return cls(env, chi_target, verbosity, **ctm_params)

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
            "_CTM_chi_target": self.chi_target,
            "_CTM_block_chi_ratio": self.block_chi_ratio,
            "_CTM_ncv_ratio": self.ncv_ratio,
            "_CTM_cutoff": self.cutoff,
            "_CTM_degen_ratio": self.degen_ratio,
        }
        data |= self._env.get_data_dic()
        np.savez_compressed(filename, **data, **additional_data)
        if self.verbosity > 0:
            print("CTMRG saved in file", filename)

    @property
    def Dmax(self):
        return self._env.Dmax

    @property
    def Dmin(self):
        return self._env.Dmin

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
    def chi_values(self):
        return self._env.chi_values

    @property
    def elementary_tensors(self):
        return self._env.elementary_tensors

    @property
    def n_sites(self):
        return self._env.n_sites

    @property
    def site_coords(self):
        return self._site_coords

    @property
    def tiling(self):
        return "\n".join("".join(s) for s in self.cell)

    def __repr__(self):
        s = f"{self.symmetry()} symmetric CTMRG with Dmax = {self.Dmax}"
        s = s + f" and chi_target = {self.chi_target}"
        return s

    def __str__(self):
        return "\n".join(
            (
                repr(self),
                f"chi values = {self.chi_values}",
                f"block_chi_ratio = {self.block_chi_ratio}",
                f"ncv_ratio = {self.ncv_ratio}",
                f"cutoff = {self.cutoff}",
                f"degen_ratio = {self.degen_ratio}",
                f"unit cell =\n{self._env.cell}",
            )
        )

    def symmetry(self):
        return self._env.symmetry()

    def get_corner_representations(self):
        return self._env.get_corner_representations()

    def restart_environment(self, dummy=True):
        """
        Restart environment tensors from elementary ones. ERASE current environment,
        use with caution.
        """
        if self.verbosity > 0:
            print("Restart brand new environment from elementary tensors.")
        tensors = [self._env.get_A(x, y) for (x, y) in self.site_coords]
        tiling = "\n".join("".join(line) for line in self.cell)
        self._env = CTM_Environment.from_elementary_tensors(tiling, tensors, dummy)
        if self.verbosity > 0:
            print(self)

    def set_symmetry(self, symmetry):
        """
        Cast all SymmetricTensor to a new symmetry.

        Parameters
        ----------
        symmetry: str
            Symmetry group
        """
        if self.symmetry() != symmetry:
            if self.verbosity > 0:
                print("set symmetry to", symmetry)
            self._env.set_symmetry(symmetry)

    def truncate_corners(self):
        """
        Truncate corners C1, C2, C3 and C4 without constructing ul, ur, dl and dr
        Use before first move to reduce corner dimension if chi_target < D^2
        """
        # we cannot approximate independently each corner with an SVD: this would
        # introduce an arbitrary unitary matrix between two corners U_1 @ V_2. To keep
        # unit cell inner compatibility, we can only renormalize bonds, not tensors.
        # So we renormalize bond between corners, without inserting edge tensors
        # basically the same thing as a standard move, without absorption.
        if self.verbosity > 0:
            print(f"Truncate corners to chi ~ {self.chi_target}")
        self.up_move_no_absorb()
        self.right_move_no_absorb()
        self.down_move_no_absorb()
        self.left_move_no_absorb()
        if self.verbosity > 2:
            self.print_tensor_shapes()

    def print_tensor_shapes(self):
        print("tensor shapes for C1 T1 C2 // T4 A T2 // C4 T3 C4:")
        for x, y in self._site_coords:
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

    def compute_rdm1x2(self, x, y):
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

    def compute_rdm2x1(self, x, y):
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

    def compute_rdm_diag_dr(self, x, y, free_memory=False):
        if self.verbosity > 1:
            print(
                f"Compute rdm for down right diagonal sites ({x+1},{y+1}) and",
                f"({x+2},{y+2})",
            )
        return rdm.rdm_diag_dr(
            self._env.get_C1(x, y),
            self._env.get_T1(x + 1, y),
            self.construct_enlarged_ur(x, y, free_memory=free_memory),
            self._env.get_T4(x, y + 1),
            self._env.get_A(x + 1, y + 1),
            self.construct_enlarged_dl(x, y, free_memory=free_memory),
            self._env.get_A(x + 2, y + 2),
            self._env.get_T2(x + 3, y + 2),
            self._env.get_T3(x + 2, y + 3),
            self._env.get_C3(x + 3, y + 3),
        )

    def compute_rdm_diag_ur(self, x, y, free_memory=False):
        if self.verbosity > 1:
            print(
                f"Compute rdm for upper right diagonal sites ({x+1},{y+2}) and",
                f"({x+2},{y+1})",
            )
        return rdm.rdm_diag_ur(
            self.construct_enlarged_ul(x, y, free_memory=free_memory),
            self._env.get_T1(x + 2, y),
            self._env.get_C2(x + 3, y),
            self._env.get_A(x + 2, y + 1),
            self._env.get_T2(x + 3, y + 1),
            self._env.get_T4(x, y + 2),
            self._env.get_A(x + 1, y + 2),
            self.construct_enlarged_dr(x, y, free_memory=free_memory),
            self._env.get_C4(x, y + 3),
            self._env.get_T3(x + 1, y + 3),
        )

    def compute_rdm2x2(self, x, y):
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
        rdm1x2_cell = []
        rdm2x1_cell = []
        for x, y in self._site_coords:
            rdm1x2_cell.append(self.compute_rdm1x2(x, y))
            rdm2x1_cell.append(self.compute_rdm2x1(x, y))
        return rdm1x2_cell, rdm2x1_cell

    def compute_rdm_2nd_neighbor_cell(self, free_memory=True):
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
        rdm_dr_cell = []
        rdm_ur_cell = []
        for x, y in self._site_coords:
            rdm_dr_cell.append(self.compute_rdm_diag_dr(x, y, free_memory=free_memory))
            rdm_ur_cell.append(self.compute_rdm_diag_ur(x, y, free_memory=free_memory))
        return rdm_dr_cell, rdm_ur_cell

    def compute_transfer_spectrum_h(
        self, nval, y=0, dmax_full=200, maxiter=1000, tol=0
    ):
        """
        Compute horizontal transfer matrix spectrum for row y.
        """
        T1s = []
        T3s = []
        for x in range(self.Lx):
            T1s.append(self._env.get_T1(x, y))
            T3s.append(self._env.get_T3(x, y + 1))
        return observables.compute_mps_transfer_spectrum(
            T1s, T3s, nval, dmax_full=dmax_full, maxiter=maxiter, tol=tol
        )

    def compute_transfer_spectrum_v(
        self, nval, x=0, dmax_full=200, maxiter=1000, tol=0
    ):
        """
        Compute vertical transfer matrix spectrum for column x.
        """
        T2s = []
        T4s = []
        for y in range(self.Ly):
            T2s.append(self._env.get_T2(x + 1, y).permute((1,), (2, 3, 0)))
            T4s.append(self._env.get_T4(x, y).permute((1, 2), (3, 0)))
        return observables.compute_mps_transfer_spectrum(
            T2s, T4s, nval, dmax_full=dmax_full, maxiter=maxiter, tol=tol
        )

    def compute_corr_length_h(self, y=0, maxiter=1000, tol=0):
        """
        Compute maximal horizontal correlation length in row between y and y+1.
        """
        v1, v2 = self.compute_transfer_spectrum_h(2, y=y, maxiter=maxiter, tol=tol)
        xi = -self.Lx / np.log(np.abs(v2))
        return xi

    def compute_corr_length_v(self, x=0, maxiter=1000, tol=0):
        """
        Compute maximal vertical correlation length in column between x and x+1.
        """
        v1, v2 = self.compute_transfer_spectrum_v(2, x=x, maxiter=maxiter, tol=tol)
        xi = -self.Ly / np.log(np.abs(v2))
        return xi

    def construct_enlarged_dr(self, x, y, free_memory=False):
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
        elif free_memory:
            self._env.set_corner_dr(x, y, None)
        return dr

    def construct_enlarged_dl(self, x, y, free_memory=False):
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
        elif free_memory:
            self._env.set_corner_dl(x, y, None)
        return dl

    def construct_enlarged_ul(self, x, y, free_memory=False):
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
        elif free_memory:
            self._env.set_corner_ul(x, y, None)
        return ul

    def construct_enlarged_ur(self, x, y, free_memory=False):
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
        elif free_memory:
            self._env.set_corner_ur(x, y, None)
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
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                self._env.get_C2(x + 2, y + 1),
            )
            self._env.store_projectors(x + 2, y, P, Pt)

        # 2) renormalize every non-equivalent C1, T1 and C2
        # need all projectors to be constructed at this time
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._site_coords:
            P = self._env.get_P(x + 1, y)
            Pt = self._env.get_Pt(x, y)
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
        for x, y in self._site_coords:
            P, Pt = construct_projectors(
                self.construct_enlarged_dl(x, y),
                self.construct_enlarged_dr(x, y, free_memory=True),
                self.construct_enlarged_ur(x, y, free_memory=True),
                self.construct_enlarged_ul(x, y),
                self.chi_target,
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                self._env.get_C3(x + 2, y + 2).transpose(),
            )
            self._env.store_projectors(x + 3, y + 2, P, Pt)

        # 2) renormalize tensors by absorbing column
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._site_coords:
            P = self._env.get_P(x, y + 1)
            Pt = self._env.get_Pt(x, y)
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
        if self.verbosity > 1:
            print("\nstart down move")
        # 1) compute isometries for every non-equivalent sites
        for x, y in self._site_coords:
            P, Pt = construct_projectors(
                self.construct_enlarged_ul(x, y),
                self.construct_enlarged_dl(x, y, free_memory=True),
                self.construct_enlarged_dr(x, y, free_memory=True),
                self.construct_enlarged_ur(x, y),
                self.chi_target,
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                self._env.get_C4(x + 1, y + 2),
            )
            self._env.store_projectors(x + 1, y + 3, P, Pt)

        # 2) renormalize every non-equivalent C3, T3 and C4
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._site_coords:
            P = self._env.get_P(x - 1, y)
            Pt = self._env.get_Pt(x, y)
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
        for x, y in self._site_coords:
            P, Pt = construct_projectors(
                self.construct_enlarged_ur(x, y),
                self.construct_enlarged_ul(x, y, free_memory=True),
                self.construct_enlarged_dl(x, y, free_memory=True),
                self.construct_enlarged_dr(x, y),
                self.chi_target,
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                self._env.get_C1(x + 1, y + 1),
            )
            self._env.store_projectors(x, y + 1, P, Pt)

        # 2) renormalize every non-equivalent C4, T4 and C1
        if self.verbosity > 2:
            print("Projectors constructed, renormalize tensors")
        for x, y in self._site_coords:
            P = self._env.get_P(x, y - 1)
            Pt = self._env.get_Pt(x, y)
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
        if self.verbosity > 1:
            print("left move completed")

    def up_move_no_absorb(self):
        # renormalize tensors without adding A-A* to reduce corner dimension
        # here renormalize bond C1-C2. Very similar to up_move, keep structure
        for x, y in self._site_coords:
            self._env.set_corner_ur(x, y, None)
            self._env.set_corner_ul(x, y, None)
            C2 = self._env.get_C2(x + 1, y)
            P, Pt = construct_projectors(
                self._env.get_C3(x + 1, y + 1).transpose(),
                C2,
                self._env.get_C1(x, y),
                self._env.get_C4(x, y + 1),
                self.chi_target,
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                C2,
            )
            self._env.store_projectors(x + 1, y, P, Pt)
        for x, y in self._site_coords:
            # in place change: read and write C1(x,y), T1(x,y), C2(x,y)
            P_trans = self._env.get_P(x + 1, y).transpose()
            Pt = self._env.get_Pt(x, y)
            nC1 = P_trans @ self._env.get_C1(x, y)
            nT1 = (P_trans @ self._env.get_T1(x, y)).permute((0, 1, 2), (3,)) @ Pt
            nT1 = nT1.permute((0,), (1, 2, 3))
            nC2 = self._env.get_C2(x, y) @ Pt
            self._env.store_renormalized_tensors(x, y, nC1, nT1, nC2)
        self._env.set_renormalized_tensors_up()

    def right_move_no_absorb(self):
        for x, y in self._site_coords:
            self._env.set_corner_ur(x, y, None)
            self._env.set_corner_dr(x, y, None)
            C3 = self._env.get_C3(x + 1, y + 1).transpose()
            P, Pt = construct_projectors(
                self._env.get_C4(x, y + 1),
                C3,
                self._env.get_C2(x + 1, y),
                self._env.get_C1(x, y),
                self.chi_target,
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                C3,
            )
            self._env.store_projectors(x + 1, y + 1, P, Pt)
        for x, y in self._site_coords:
            P = self._env.get_P(x, y + 1)
            Pt_trans = self._env.get_Pt(x, y).transpose()
            nC2 = P.transpose() @ self._env.get_C2(x, y)
            nT2 = (Pt_trans @ self._env.get_T2(x, y)).permute((0, 2, 3), (1,)) @ P
            nT2 = nT2.permute((0,), (3, 1, 2))
            nC3 = Pt_trans @ self._env.get_C3(x, y)
            self._env.store_renormalized_tensors(x, y, nC2, nT2, nC3)
        self._env.set_renormalized_tensors_right()

    def down_move_no_absorb(self):
        for x, y in self._site_coords:
            self._env.set_corner_dl(x, y, None)
            self._env.set_corner_dr(x, y, None)
            C4 = self._env.get_C4(x, y + 1)
            P, Pt = construct_projectors(
                self._env.get_C1(x, y),
                C4,
                self._env.get_C3(x + 1, y + 1).transpose(),
                self._env.get_C2(x + 1, y),
                self.chi_target,
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                C4,
            )
            self._env.store_projectors(x, y + 1, P, Pt)
        for x, y in self._site_coords:
            P = self._env.get_P(x - 1, y)
            Pt = self._env.get_Pt(x, y)
            nC3 = self._env.get_C3(x, y) @ P
            nT3 = (self._env.get_T3(x, y) @ P).permute((0, 1, 3), (2,)) @ Pt
            nT3 = nT3.permute((0, 1, 3), (2,))
            nC4 = self._env.get_C4(x, y) @ Pt
            self._env.store_renormalized_tensors(x, y, nC3, nT3, nC4)
        self._env.set_renormalized_tensors_down()

    def left_move_no_absorb(self):
        for x, y in self._site_coords:
            self._env.set_corner_ul(x, y, None)
            self._env.set_corner_dl(x, y, None)
            C1 = self._env.get_C1(x, y)
            P, Pt = construct_projectors(
                self._env.get_C2(x + 1, y),
                C1,
                self._env.get_C4(x, y + 1),
                self._env.get_C3(x + 1, y + 1).transpose(),
                self.chi_target,
                self.block_chi_ratio,
                self.ncv_ratio,
                self.cutoff,
                self.degen_ratio,
                C1,
            )
            self._env.store_projectors(x, y, P, Pt)
        for x, y in self._site_coords:
            P_trans = self._env.get_P(x, y - 1).transpose()
            Pt = self._env.get_Pt(x, y)
            nC4 = P_trans @ self._env.get_C4(x, y)
            nT4 = (P_trans @ self._env.get_T4(x, y)).permute((0, 1, 2), (3,)) @ Pt
            nT4 = nT4.permute((0,), (1, 2, 3))
            nC1 = self._env.get_C1(x, y) @ Pt
            self._env.store_renormalized_tensors(x, y, nC4, nT4, nC1)
        self._env.set_renormalized_tensors_left()
