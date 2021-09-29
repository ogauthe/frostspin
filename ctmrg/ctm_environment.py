import numpy as np

from symmetric_tensor.u1_symmetric_tensor import U1_SymmetricTensor

# TODO upgrade savefile format


def _initialize_env(A):
    #
    #   C1-0   3-T1-0   1-C2
    #   |        ||        |
    #   1        12        0
    #
    #   0       0 2        0
    #   |        \|        |
    #   T4=1   5--A--3  2=T2
    #   |  2     /|     3  |
    #   3       1 4        1
    #
    #   0        01        0
    #   |        ||        |
    #   C4-1   3-T3-2   1-C3

    # merge axes quick and dirty
    combine = A.combine_representations
    axes = A.axis_reps
    caxes = tuple(A.conjugate_representation(r) for r in axes)

    def init_C(At):
        temp = At.H @ At
        temp = temp.permutate((2, 0), (3, 1))
        row_rep = combine(*temp.axis_reps[:2])
        col_rep = combine(*temp.axis_reps[2:])
        C = type(At)((row_rep, col_rep), 1, temp.blocks, temp.block_irreps)
        return C

    C1 = init_C(A.permutate((0, 1, 2, 5), (3, 4)))
    C2 = init_C(A.permutate((0, 1, 2, 3), (4, 5)))
    C3 = init_C(A.permutate((0, 1, 3, 4), (2, 5)))
    C4 = init_C(A.permutate((0, 1, 4, 5), (2, 3)))

    temp = A.permutate((0, 1, 2), (3, 4, 5))
    temp = temp.H @ temp
    temp = temp.permutate((3, 0), (4, 1, 5, 2))
    repT1 = (combine(caxes[3], axes[3]), axes[4], caxes[4], combine(axes[5], caxes[5]))
    T1 = type(A)(repT1, 1, temp.blocks, temp.block_irreps)

    temp = A.permutate((0, 1, 3), (2, 4, 5))
    temp = temp.H @ temp
    temp = temp.permutate((3, 0), (4, 1, 5, 2))
    repT2 = (combine(caxes[2], axes[2]), combine(axes[4], caxes[4]), axes[5], caxes[5])
    T2 = type(A)(repT2, 1, temp.blocks, temp.block_irreps)

    temp = A.permutate((0, 1, 4), (2, 3, 5))
    temp = temp.H @ temp
    temp = temp.permutate((3, 0, 4, 1), (5, 2))
    repT3 = (caxes[2], axes[2], combine(caxes[3], axes[3]), combine(axes[5], caxes[5]))
    T3 = type(A)(repT3, 3, temp.blocks, temp.block_irreps)

    temp = A.permutate((0, 1, 5), (2, 3, 4))
    temp = temp.H @ temp
    temp = temp.permutate((3, 0), (4, 1, 5, 2))
    repT4 = (combine(caxes[2], axes[2]), axes[3], caxes[3], combine(axes[4], caxes[4]))
    T4 = type(A)(repT4, 1, temp.blocks, temp.block_irreps)

    return C1, T1, C2, T2, C3, T3, C4, T4


def _block_AAconj(A):
    """
    Construct U1_SymmetricTensor versions of double layer tensor a = A-A* that can be
    used in add_a_bilayer. One version for each corner is needed, therefore we have
    a_ul, a_ur, a_dl and a_dr, still we can define a_dl = a_ur.T and a_dr = a_ul.TH to
    save memory (same block memory).
    To be able to use a_ur and a_ul in the same function, non-clockwise leg order is
    required in a_ur.

        45                       67
        ||                       ||
    67=a_ul=01               23=a_ur=45
        ||                       ||
        23                       01

        23                       01
        ||                       ||
    67=a_dl=01 = a_ur.T       23=a_dr=45 = a_ul.T
        ||                       ||
        45                       67
    """
    # optimize dot(A,A*) using A.H, then put conjugate version as 2nd layer in permutate
    a_ul = A.H @ A
    a_ul = a_ul.permutate((5, 1, 6, 2), (4, 0, 7, 3))
    a_rd = a_ul.T

    a_ur = a_ul.permutate((2, 3, 6, 7), (0, 1, 4, 5))
    a_dl = a_ur.T
    return a_ul, a_ur, a_rd, a_dl


class CTM_Environment(object):
    """
    Container for CTMRG environment tensors. Follow leg conventions from CTMRG.
    """

    def __init__(self, cell, neq_As, load_env=None):
        """
        Initialize environment tensors. Consider calling from_elementary_tensors
        or from_file instead of directly calling this function.
        """
        # this function makes no input validation, needs to be done before

        # 1) Define indices and neq_coords from cell
        # construct list of unique letters sorted according to appearance order in cell
        # (may be different from lexicographic order)
        seen = set()
        seen_add = seen.add
        letters = [c for c in cell.flat if not (c in seen or seen_add(c))]
        self._Nneq = len(letters)

        # [row,col] indices are transposed from (x,y) coordinates but (x,y) is natural
        # to specify positions, so we need to transpose indices here to get simple CTMRG
        # code. Construct indices and neq_coords such that
        # - for all i in range(Nneq), i == indices[neq_coords[i][0], neq_coords[i][1]]
        # - for all (x,y) in neq_coords, (x,y) == neq_coords[indices[x,y]]
        self._neq_coords = np.empty((self._Nneq, 2), dtype=np.int8)
        indices = np.empty(cell.shape, dtype=int)
        for i, l in enumerate(letters):
            inds_l = cell == l  # a tensor can appear more than once in tiling
            ind_values = inds_l.nonzero()
            self._neq_coords[i] = ind_values[1][0], ind_values[0][0]  # transpose
            indices[inds_l] = i

        self._Ly, self._Lx = cell.shape
        self._cell = cell
        self._indices = indices.T.copy()  # transpose

        # 2) store tensors. They have to follow cell.flat order.
        self._neq_As = neq_As
        self._Dmax = max(max(A.shape[2:]) for A in self._neq_As)

        # 3) Construct double layer tensor A-A* for use in add_a_block
        self._a_ur, self._a_ul, self._a_rd, self._a_dl = [[] for i in range(4)]
        for A in neq_As:
            a_ul, a_ur, a_rd, a_dl = _block_AAconj(A)
            self._a_ur.append(a_ur)
            self._a_ul.append(a_ul)
            self._a_rd.append(a_rd)
            self._a_dl.append(a_dl)

        # 4) initialize environment tensors
        if load_env is None:
            self.restart()
        else:  # from file
            loaded = self.load_environment_from_file(load_env)
            if not loaded:
                self.restart()

    @property
    def Dmax(self):
        return self._Dmax

    @property
    def chi_max(self):
        return max(
            max(C.shape)
            for C in self._neq_C1s + self._neq_C2s + self._neq_C3s + self._neq_C4s
        )

    def restart(self):
        """
        Erase current environment tensors C and T and restart them from elementary
        tensors.
        """
        self._neq_C1s, self._neq_T1s = [], []
        self._neq_C2s, self._neq_T2s = [], []
        self._neq_C3s, self._neq_T3s = [], []
        self._neq_C4s, self._neq_T4s = [], []
        for A in self._neq_As:
            C1, T1, C2, T2, C3, T3, C4, T4 = _initialize_env(A)
            self._neq_C1s.append(C1)
            self._neq_T1s.append(T1)
            self._neq_C2s.append(C2)
            self._neq_T2s.append(T2)
            self._neq_C3s.append(C3)
            self._neq_T3s.append(T3)
            self._neq_C4s.append(C4)
            self._neq_T4s.append(T4)

        self._corners_ul = [None] * self._Nneq
        self._corners_ur = [None] * self._Nneq
        self._corners_dl = [None] * self._Nneq
        self._corners_dr = [None] * self._Nneq
        self._reset_temp_lists()

    @classmethod
    def from_elementary_tensors(cls, tiling, tensors, representations, load_env=None):
        """
        Construct CTM_Environment from elementary tensors according to tiling.

        Parameters
        ----------
        tiling : string
            Tiling pattern, such as 'AB\nBA' or 'AB\nCD'.
        tensors : iterable of Nneq numpy arrays
            Tensors given from left to right and up to down (as in array.flat)
        representations : enum of Nneq representation tuple
            Representation for each leg of each tensor.
        load_env : string
            File containing previously computed environment to restart from.
        """
        txt = [" ".join(w) for w in tiling.strip().splitlines()]
        cell = np.atleast_2d(np.genfromtxt(txt, dtype="U1"))
        Nneq = len(set(cell.flat))
        if len(tensors) != Nneq:
            raise ValueError("Tensor number do not match tiling")
        if len(representations) != Nneq:
            raise ValueError("Representation number do not match tiling")

        # store tensors according to cell.flat order
        neq_As = []
        for A0, rep_A in zip(tensors, representations):
            if A0.ndim != 6:
                raise ValueError("Elementary tensor must be of rank 6")
            neq_As.append(U1_SymmetricTensor.from_array(A0, rep_A, 2))

        return cls(cell, neq_As, load_env=load_env)

    def get_data_to_save(self):
        """
        Return environment data as a dict to save in external file.
        """
        # do not store lists to avoid pickle
        # come back to elementary numpy arrays
        data = {"_CTM_cell": self._cell}

        # save dense array + add minus signs for backward compatibility
        for i in range(self._Nneq):
            data[f"_CTM_A_{i}"] = self._neq_As[i].toarray()
            data[f"_CTM_C1_{i}"] = self._neq_C1s[i].toarray()
            data[f"_CTM_T1_{i}"] = self._neq_T1s[i].toarray()
            data[f"_CTM_C2_{i}"] = self._neq_C2s[i].toarray()
            data[f"_CTM_T2_{i}"] = self._neq_T2s[i].toarray()
            data[f"_CTM_C3_{i}"] = self._neq_C3s[i].toarray()
            data[f"_CTM_T3_{i}"] = self._neq_T3s[i].toarray()
            data[f"_CTM_C4_{i}"] = self._neq_C4s[i].toarray()
            data[f"_CTM_T4_{i}"] = self._neq_T4s[i].toarray()
            data[f"_CTM_colors_C1_r_{i}"] = self._neq_C1s[i].axis_reps[0]
            data[f"_CTM_colors_C1_d_{i}"] = -self._neq_C1s[i].axis_reps[1]
            data[f"_CTM_colors_C2_d_{i}"] = self._neq_C2s[i].axis_reps[0]
            data[f"_CTM_colors_C2_l_{i}"] = -self._neq_C2s[i].axis_reps[1]
            data[f"_CTM_colors_C3_u_{i}"] = self._neq_C3s[i].axis_reps[0]
            data[f"_CTM_colors_C3_l_{i}"] = -self._neq_C3s[i].axis_reps[1]
            data[f"_CTM_colors_C4_u_{i}"] = self._neq_C4s[i].axis_reps[0]
            data[f"_CTM_colors_C4_r_{i}"] = -self._neq_C4s[i].axis_reps[1]

            # legacy: A must be save with conjugate_columns = True
            data[f"_CTM_colors_A_{i}_0"] = self._neq_As[i].axis_reps[0]
            data[f"_CTM_colors_A_{i}_1"] = self._neq_As[i].axis_reps[1]
            for leg in range(2, 6):
                data[f"_CTM_colors_A_{i}_{leg}"] = -self._neq_As[i].axis_reps[leg]

        return data

    @classmethod
    def from_file(cls, savefile):
        """
        Construct CTM_Environment from save file.
        """
        # 2 steps construction: first load elementary tensors
        # then initialize object to be able to navigate through unit cell
        # finally reopen file to load environment: calling C(x1, y1) representations to
        # construct T(x2, y2). This is not possible with just flat indices.
        # TODO change this (requires change in savefile format to save T representation)

        neq_As, reps_A = [], []
        with np.load(savefile) as data:
            cell = data["_CTM_cell"]
            Nneq = len(set(cell.flat))
            for i in range(Nneq):
                neq_As.append(data[f"_CTM_A_{i}"])
                reps = tuple(data[f"_CTM_colors_A_{i}_{leg}"] for leg in range(6))
                reps_A.append(reps)

        # call from array after closing file
        for i in range(Nneq):
            neq_As[i] = U1_SymmetricTensor.from_array(neq_As[i], reps_A[i], 2)

        return cls(cell, neq_As, load_env=savefile)

    def load_environment_from_file(self, savefile):
        """
        Load environment tensors from save file.
        If representations do not match current elementary tensor representations, no
        change is made on self and False is returned. Else environment tensors are
        loaded and the function returns True.
        """
        # TODO upgrade savefile format
        neq_C1s, neq_C2s, neq_C3s, neq_C4s = [[] for i in range(4)]
        neq_T1s, neq_T2s, neq_T3s, neq_T4s = [[] for i in range(4)]
        r1r, r1d, r2d, r2l, r3u, r3l, r4u, r4r = [[] for i in range(8)]

        with np.load(savefile) as data:
            if (self._cell != data["_CTM_cell"]).any():
                print(
                    " *** WARNING *** no CTM environment reload:",
                    "unit cells do not match",
                )
                return False

            # load dense array + add minus signs for backward compatibility
            for i in range(self._Nneq):
                saxes = [sorted(r) for r in self._neq_As[i].axis_reps]
                match = sorted(data[f"_CTM_colors_A_{i}_0"]) == saxes[0]
                match &= sorted(data[f"_CTM_colors_A_{i}_1"]) == saxes[1]
                for leg in range(2, 6):
                    match &= sorted(-data[f"_CTM_colors_A_{i}_{leg}"]) == saxes[leg]
                if not match:
                    print(
                        " *** WARNING *** no CTM environment reload:",
                        "representations do not match",
                    )
                    return False

                neq_C1s.append(data[f"_CTM_C1_{i}"])
                neq_C2s.append(data[f"_CTM_C2_{i}"])
                neq_C3s.append(data[f"_CTM_C3_{i}"])
                neq_C4s.append(data[f"_CTM_C4_{i}"])
                neq_T1s.append(data[f"_CTM_T1_{i}"])
                neq_T2s.append(data[f"_CTM_T2_{i}"])
                neq_T3s.append(data[f"_CTM_T3_{i}"])
                neq_T4s.append(data[f"_CTM_T4_{i}"])
                r1r.append(data[f"_CTM_colors_C1_r_{i}"])
                r1d.append(-data[f"_CTM_colors_C1_d_{i}"])
                r2d.append(data[f"_CTM_colors_C2_d_{i}"])
                r2l.append(-data[f"_CTM_colors_C2_l_{i}"])
                r3u.append(data[f"_CTM_colors_C3_u_{i}"])
                r3l.append(-data[f"_CTM_colors_C3_l_{i}"])
                r4u.append(data[f"_CTM_colors_C4_u_{i}"])
                r4r.append(-data[f"_CTM_colors_C4_r_{i}"])

        # call from array after closing file
        for i in range(self._Nneq):
            neq_C1s[i] = U1_SymmetricTensor.from_array(
                neq_C1s[i], (r1r[i], r1d[i]), 1, conjugate_columns=False
            )
            neq_C2s[i] = U1_SymmetricTensor.from_array(
                neq_C2s[i], (r2d[i], r2l[i]), 1, conjugate_columns=False
            )
            neq_C3s[i] = U1_SymmetricTensor.from_array(
                neq_C3s[i], (r3u[i], r3l[i]), 1, conjugate_columns=False
            )
            neq_C4s[i] = U1_SymmetricTensor.from_array(
                neq_C4s[i], (r4u[i], r4r[i]), 1, conjugate_columns=False
            )

        # first fill corners, to access their representation with get_Ci(x,y)
        self._neq_C1s = neq_C1s
        self._neq_C2s = neq_C2s
        self._neq_C3s = neq_C3s
        self._neq_C4s = neq_C4s

        for i, (x, y) in enumerate(self._neq_coords):
            axes = self._a_ul[i].axis_reps

            r1r = self.get_C1(x - 1, y).axis_reps[0]
            r2l = self.get_C2(x + 1, y).axis_reps[1]
            repsT1 = (r2l, -axes[2], -axes[3], r1r)
            neq_T1s[i] = U1_SymmetricTensor.from_array(
                neq_T1s[i], repsT1, 1, conjugate_columns=False
            )

            r2d = -self.get_C2(x, y + 1).axis_reps[0]
            r3u = self.get_C3(x, y - 1).axis_reps[0]
            repsT2 = (r2d, r3u, axes[6], axes[7])
            neq_T2s[i] = U1_SymmetricTensor.from_array(
                neq_T2s[i], repsT2, 1, conjugate_columns=False
            )

            r3l = self.get_C3(x + 1, y).axis_reps[1]
            r4r = -self.get_C4(x - 1, y).axis_reps[1]
            repsT3 = (-axes[4], -axes[5], r3l, r4r)
            neq_T3s[i] = U1_SymmetricTensor.from_array(
                neq_T3s[i], repsT3, 3, conjugate_columns=False
            )

            r4u = -self.get_C4(x, y - 1).axis_reps[0]
            r1d = self.get_C1(x, y + 1).axis_reps[1]
            repsT4 = (r1d, -axes[0], -axes[1], -r4u)
            neq_T4s[i] = U1_SymmetricTensor.from_array(
                neq_T4s[i], repsT4, 1, conjugate_columns=False
            )

        self._neq_T1s = neq_T1s
        self._neq_T2s = neq_T2s
        self._neq_T3s = neq_T3s
        self._neq_T4s = neq_T4s

        # reset constructed corners
        self._corners_ul = [None] * self._Nneq
        self._corners_ur = [None] * self._Nneq
        self._corners_dl = [None] * self._Nneq
        self._corners_dr = [None] * self._Nneq
        self._reset_temp_lists()
        return True

    def set_tensors(self, tensors, representations):
        """
        Set new elementary tensors while keeping environment tensors if possible.
        """
        if len(tensors) != self._Nneq:
            raise ValueError("Incompatible cell and tensors")
        if len(representations) != self._Nneq:
            raise ValueError("Incompatible cell and representations")

        restart_env = False
        for i, A0 in enumerate(tensors):
            if A0.ndim != 6:
                raise ValueError("Elementary tensor must be of rank 6")
            A = U1_SymmetricTensor.from_array(A0, representations[i], 2)
            # permutation of irreps inside an abelian representation are allowed: irrep
            # blocks will not be affected.
            # However if some sector size changes, cannot keep env: restart from scratch
            # use lists to catch total dimension change
            for r1, r2 in zip(A.axis_reps, self._neq_As[i].axis_reps):
                if sorted(r1) != sorted(r2):
                    restart_env = True

            a_ul, a_ur, a_rd, a_dl = _block_AAconj(A)
            self._neq_As[i] = A
            self._a_ur[i] = a_ur
            self._a_ul[i] = a_ul
            self._a_rd[i] = a_rd
            self._a_dl[i] = a_dl

        if restart_env:
            print("*** WARNING *** restart environment from scratch")
            self.restart()
            self._Dmax = max(max(A.shape[2:]) for A in self._neq_As)

        else:  # reset all corners C-T // T-A since A changed
            self._corners_ul = [None] * self._Nneq
            self._corners_ur = [None] * self._Nneq
            self._corners_dl = [None] * self._Nneq
            self._corners_dr = [None] * self._Nneq

    @property
    def cell(self):
        return self._cell

    @property
    def Nneq(self):
        return self._Nneq

    @property
    def Lx(self):
        return self._Lx

    @property
    def Ly(self):
        return self._Ly

    @property
    def neq_coords(self):
        return self._neq_coords

    def get_tensor_type(self, x, y):
        return self._cell[x % self._Lx, y % self._Ly]

    def get_A(self, x, y):
        return self._neq_As[self._indices[x % self._Lx, y % self._Ly]]

    def get_C1(self, x, y):
        return self._neq_C1s[self._indices[x % self._Lx, y % self._Ly]]

    def get_T1(self, x, y):
        return self._neq_T1s[self._indices[x % self._Lx, y % self._Ly]]

    def get_C2(self, x, y):
        return self._neq_C2s[self._indices[x % self._Lx, y % self._Ly]]

    def get_T2(self, x, y):
        return self._neq_T2s[self._indices[x % self._Lx, y % self._Ly]]

    def get_C3(self, x, y):
        return self._neq_C3s[self._indices[x % self._Lx, y % self._Ly]]

    def get_T3(self, x, y):
        return self._neq_T3s[self._indices[x % self._Lx, y % self._Ly]]

    def get_C4(self, x, y):
        return self._neq_C4s[self._indices[x % self._Lx, y % self._Ly]]

    def get_T4(self, x, y):
        return self._neq_T4s[self._indices[x % self._Lx, y % self._Ly]]

    def get_P(self, x, y):
        return self._neq_P[self._indices[x % self._Lx, y % self._Ly]]

    def get_Pt(self, x, y):
        return self._neq_Pt[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_ul(self, x, y):
        return self._a_ul[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_ur(self, x, y):
        return self._a_ur[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_rd(self, x, y):
        return self._a_rd[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_dl(self, x, y):
        return self._a_dl[self._indices[x % self._Lx, y % self._Ly]]

    def get_corner_ul(self, x, y):
        return self._corners_ul[self._indices[x % self._Lx, y % self._Ly]]

    def get_corner_ur(self, x, y):
        return self._corners_ur[self._indices[x % self._Lx, y % self._Ly]]

    def get_corner_dl(self, x, y):
        return self._corners_dl[self._indices[x % self._Lx, y % self._Ly]]

    def get_corner_dr(self, x, y):
        return self._corners_dr[self._indices[x % self._Lx, y % self._Ly]]

    def set_corner_ul(self, x, y, ul):
        self._corners_ul[self._indices[x % self._Lx, y % self._Ly]] = ul

    def set_corner_ur(self, x, y, ur):
        self._corners_ur[self._indices[x % self._Lx, y % self._Ly]] = ur

    def set_corner_dl(self, x, y, dl):
        self._corners_dl[self._indices[x % self._Lx, y % self._Ly]] = dl

    def set_corner_dr(self, x, y, dr):
        self._corners_dr[self._indices[x % self._Lx, y % self._Ly]] = dr

    def store_projectors(self, x, y, P, Pt):
        j = self._indices[x % self._Lx, y % self._Ly]
        self._neq_P[j] = P
        self._neq_Pt[j] = Pt

    def store_renormalized_tensors(self, x, y, nCX, nT, nCY):
        j = self._indices[x % self._Lx, y % self._Ly]
        self._nCX[j] = nCX
        self._nT[j] = nT
        self._nCY[j] = nCY

    def _reset_temp_lists(self):
        # reset is needed because no list copy occurs
        self._neq_P = [None] * self._Nneq
        self._neq_Pt = [None] * self._Nneq
        self._nCX = [None] * self._Nneq
        self._nT = [None] * self._Nneq
        self._nCY = [None] * self._Nneq

    def set_renormalized_tensors_up(self):
        self._neq_C1s = self._nCX
        self._neq_T1s = self._nT
        self._neq_C2s = self._nCY
        self._corners_ul = [None] * self._Nneq
        self._corners_ur = [None] * self._Nneq
        self._reset_temp_lists()

    def set_renormalized_tensors_right(self):
        self._neq_C2s = self._nCX
        self._neq_T2s = self._nT
        self._neq_C3s = self._nCY
        self._corners_ur = [None] * self._Nneq
        self._corners_dr = [None] * self._Nneq
        self._reset_temp_lists()

    def set_renormalized_tensors_down(self):
        self._neq_C3s = self._nCX
        self._neq_T3s = self._nT
        self._neq_C4s = self._nCY
        self._corners_dr = [None] * self._Nneq
        self._corners_dl = [None] * self._Nneq
        self._reset_temp_lists()

    def set_renormalized_tensors_left(self):
        self._neq_C4s = self._nCX
        self._neq_T4s = self._nT
        self._neq_C1s = self._nCY
        self._corners_dl = [None] * self._Nneq
        self._corners_ul = [None] * self._Nneq
        self._reset_temp_lists()
