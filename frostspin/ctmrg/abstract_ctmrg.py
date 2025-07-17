import numpy as np

from . import observables, rdm
from .ctm_contract import (
    contract_C1234,
    contract_dl,
    contract_dr,
    contract_norm,
    contract_T1T3,
    contract_T2T4,
    contract_ul,
    contract_ur,
)
from .ctm_environment import CTMEnvironment


class AbstractCTMRG:
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

    The environment tensors are stored in a custom CTMEnvironment class to easily deal
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

    def __init__(self, env, chi_target, *, verbosity=0, **ctm_params):
        """
        Constructor for totally asymmetric CTMRG algorithm. Consider using from_file or
        from_elementary_tensors methods instead of calling this one directly.

        Parameters
        ----------
        env: CTMEnvironment
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

        if type(self) is AbstractCTMRG:
            raise NotImplementedError("Cannot instantiate AbstractCTMRG")

        self._env = env
        self._site_coords = env.site_coords
        self.chi_target = int(chi_target)

        # set parameters.
        self.block_chi_ratio = float(ctm_params.get("block_chi_ratio", 1.1))
        self.ncv_ratio = float(ctm_params.get("ncv_ratio", 2.0))
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

    @classmethod
    def from_elementary_tensors(
        cls, tiling, tensors, chi_target, *, verbosity=0, **ctm_params
    ):
        """
        Construct CTMRG from elementary tensors and tiling.
        Environment tensors are initialized as dummy and act as identity between bra and
        ket layers: corners are (1x1) identity matrices with trivial representation and
        edges are identiy matrices between layers with a representation matching site
        tensor, with dummy legs added to match corners.

        Parameters
        ----------
        tensors : enumerable of SymmetricTensor
            Elementary tensors of unit cell, from left to right from top to bottom.
        tiling : string
            String defining the shape of the unit cell, typically "A" or "AB\nCD".
        chi_target : integer
            Target for corner dimension. This is a target, actual corner dimension chi
            may differ differ independently on a any corner due to cutoff or multiplets.
        verbosity : int
            Level of log verbosity. Default is no log.

        See `__init__` for a description of additional parameters.
        """
        if verbosity > 0:
            print("Start CTMRG from dummy environment")
        env = CTMEnvironment.from_elementary_tensors(tiling, tensors)
        return cls(env, chi_target, verbosity=verbosity, **ctm_params)

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
    def load_from_file(cls, filename, *, verbosity=0):
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
        env = CTMEnvironment.load_from_file(filename)
        return cls(env, chi_target, verbosity=verbosity, **ctm_params)

    def save_to_file(self, filename, **additional_data):
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
        return s + f" and chi_target = {self.chi_target}"

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

    def get_A(self, x, y):
        return self._env.get_A(x, y)

    def get_C1(self, x, y):
        return self._env.get_C1(x, y)

    def get_C2(self, x, y):
        return self._env.get_C2(x, y)

    def get_C3(self, x, y):
        return self._env.get_C3(x, y)

    def get_C4(self, x, y):
        return self._env.get_C4(x, y)

    def get_T1(self, x, y):
        return self._env.get_T1(x, y)

    def get_T2(self, x, y):
        return self._env.get_T2(x, y)

    def get_T3(self, x, y):
        return self._env.get_T3(x, y)

    def get_T4(self, x, y):
        return self._env.get_T4(x, y)

    def get_corner_representations(self):
        return self._env.get_corner_representations()

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

    def compute_PEPS_norm_log(self):
        """
        Compute logarithm of PEPS norm per site.
        """
        # TODO how to deal with complex?
        logZ = 0.0
        for x, y in self._site_coords:
            nc = contract_C1234(
                self._env.get_C1(x, y),
                self._env.get_C2(x + 1, y),
                self._env.get_C4(x, y + 1),
                self._env.get_C3(x + 1, y + 1),
            )
            nt13 = contract_T1T3(
                self._env.get_C1(x, y),
                self._env.get_T1(x + 1, y),
                self._env.get_C2(x + 2, y),
                self._env.get_C4(x, y + 1),
                self._env.get_T3(x + 1, y + 1),
                self._env.get_C3(x + 2, y + 1),
            )
            nt24 = contract_T2T4(
                self._env.get_C1(x, y),
                self._env.get_C2(x + 1, y),
                self._env.get_T4(x, y + 1),
                self._env.get_T2(x + 1, y + 1),
                self._env.get_C4(x, y + 2),
                self._env.get_C3(x + 1, y + 2),
            )
            n1234 = contract_norm(
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
            if nc * nt13 * nt24 * n1234 < 0:
                print("Warning: negative PEPS norm")
                return np.nan
            logZ += (
                np.log(np.abs(n1234))
                + np.log(np.abs(nc))
                - np.log(np.abs(nt13))
                - np.log(np.abs(nt24))
            )
        logZ /= self.n_sites
        return logZ

    def compute_rdm1x1(self, x, y):
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

    def compute_rdm_diag_dr(self, x, y, *, free_memory=False):
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

    def compute_rdm_diag_ur(self, x, y, *, free_memory=False):
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

    def compute_rdm1x1_cell(self):
        """
        Compute reduced density matrix for every single site in the unit cell
        """
        rdm1x1_cell = []
        for x, y in self._site_coords:
            rdm1x1_cell.append(self.compute_rdm1x1(x, y))
        return rdm1x1_cell

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
        rdm_dr_cell = []
        rdm_ur_cell = []
        for x, y in self._site_coords:
            rdm_dr_cell.append(self.compute_rdm_diag_dr(x, y, free_memory=free_memory))
            rdm_ur_cell.append(self.compute_rdm_diag_ur(x, y, free_memory=free_memory))
        return rdm_dr_cell, rdm_ur_cell

    def compute_transfer_spectrum_h(
        self, y, nvals, *, dmax_full=200, maxiter=1000, arpack_tol=0
    ):
        """
        Compute nval (dense) largest eigenvalues of the horizontal transfer matrix at
        unit cell row y.
        """
        T1s = []
        T3s = []
        for x in range(self.Lx):
            T1s.append(self._env.get_T1(-x, y))
            T3s.append(self._env.get_T3(-x, y + 1))
        return observables.compute_mps_transfer_spectrum(
            T1s,
            T3s,
            nvals,
            dmax_full=dmax_full,
            maxiter=maxiter,
            arpack_tol=arpack_tol,
            verbosity=self.verbosity,
        )

    def compute_transfer_spectrum_v(
        self, x, nvals, *, dmax_full=200, maxiter=1000, arpack_tol=0
    ):
        """
        Compute nval (dense) largest eigenvalues of the vertical transfer matrix at
        unit cell column x.
        """
        T1s = []
        T3s = []
        for y in range(self.Ly):
            T1s.append(self._env.get_T2(x + 1, -y).permute((1,), (2, 3, 0)))
            T3s.append(self._env.get_T4(x, -y).permute((1, 2), (3, 0)))
        return observables.compute_mps_transfer_spectrum(
            T1s,
            T3s,
            nvals,
            dmax_full=dmax_full,
            maxiter=maxiter,
            arpack_tol=arpack_tol,
            verbosity=self.verbosity,
        )

    def compute_corr_length_h(self, y, *, maxiter=1000, arpack_tol=0):
        """
        Compute maximal vertical correlation length at unit cell row y.
        """
        v1, v2 = self.compute_transfer_spectrum_h(
            y, 2, maxiter=maxiter, arpack_tol=arpack_tol
        )
        return -self.Lx / np.log(np.abs(v2))

    def compute_corr_length_v(self, x, *, maxiter=1000, arpack_tol=0):
        """
        Compute maximal vertical correlation length at unit cell column x.
        """
        v1, v2 = self.compute_transfer_spectrum_v(
            x, 2, maxiter=maxiter, arpack_tol=arpack_tol
        )
        return -self.Ly / np.log(np.abs(v2))

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
        elif free_memory:
            self._env.set_corner_dr(x, y, None)
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
        elif free_memory:
            self._env.set_corner_dl(x, y, None)
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
        elif free_memory:
            self._env.set_corner_ul(x, y, None)
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
        elif free_memory:
            self._env.set_corner_ur(x, y, None)
        return ur
