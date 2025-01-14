import numpy as np

from frostspin.symmetric_tensor.tools import get_symmetric_tensor_type


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

    # leg ordering is set according to diagram upside
    # all functions using edge T use some transpose to fix n_row_reps
    # so there can be no error associated with it
    # however signature matter.
    # convention: corners C1, C2 and C4 have standard matrix signature ->-C->-
    # ie 0 for row and 1 for column
    # it is not possible to impose the same convention for C3
    # edges signature for corner legs have to obey
    # this convention does not change through iteration. (In any case SVD always
    # produces isometries with matching signatures)
    # for edges legs merging with center A, signature must respect A.signature
    # no assumption is made on it.

    def init_C(At, su):
        C = At.dagger() @ At
        C = C.permute((2, 0), (3, 1))
        C.update_signature(su)
        return C.merge_legs(2, 3).merge_legs(0, 1)

    # Due to C3 annoying conventions, some -1 need to appear here. These signature
    # update are independent of A signature or of the unit cell tiling pattern.
    C1 = init_C(A.permute((0, 1, 2, 5), (3, 4)), [0, 1, 0, 1])
    C2 = init_C(A.permute((0, 1, 2, 3), (4, 5)), [0, -1, 0, 1])
    C3 = init_C(A.permute((0, 1, 3, 4), (2, 5)), [0, 1, 0, 1])
    C4 = init_C(A.permute((0, 1, 4, 5), (2, 3)), [0, 1, 0, -1])

    # merging is trivial for abelian symmetries, however for non-abelian symmetries
    # one must be careful not to change tree structure. With current tree structure
    #       singlets
    #       /     /
    #      /\    /\
    #     /\ \  /\ \
    #
    # only first 2 legs of either rows or columns can be merged: merging legs 1 and 2
    # creates different tree strucutre
    #    /
    #   /\
    #  / /\
    #
    # to avoid this, additional leg permutations are added so that legs to be merged are
    # always 0 and 1 in row or columns. Additional cost is very low, just add one
    # permutation on T with total size D^6.

    temp = A.permute((0, 1, 2), (3, 4, 5))
    T1 = temp.dagger() @ temp
    T1 = T1.permute((3, 0), (5, 2, 4, 1))
    T1.update_signature([0, 1, 0, 1, 0, 0])
    T1 = T1.merge_legs(2, 3).merge_legs(0, 1)
    T1 = T1.permute((0,), (2, 3, 1))

    temp = A.permute((0, 1, 3), (2, 4, 5))
    T2 = temp.dagger() @ temp
    T2 = T2.permute((3, 0), (4, 1, 5, 2))
    T2.update_signature([0, 1, 0, 1, 0, 0])
    T2 = T2.merge_legs(2, 3).merge_legs(0, 1)

    temp = A.permute((0, 1, 4), (2, 3, 5))
    T3 = temp.dagger() @ temp
    T3 = T3.permute((4, 1), (5, 2, 3, 0))
    T3.update_signature([0, 1, 0, 1, 0, 0])
    T3 = T3.merge_legs(2, 3).merge_legs(0, 1)
    T3 = T3.permute((2, 3, 0), (1,))

    temp = A.permute((0, 1, 5), (2, 3, 4))
    T4 = temp.dagger() @ temp
    T4 = T4.permute((3, 0), (5, 2, 4, 1))
    T4.update_signature([0, 1, 0, 1, 0, 0])
    T4 = T4.merge_legs(2, 3).merge_legs(0, 1)
    T4 = T4.permute((0,), (2, 3, 1))

    return C1, T1, C2, T2, C3, T3, C4, T4


def _initialize_dummy(A):
    ST = type(A)

    rr = (ST.singlet(),)
    C1 = ST.from_array(np.eye(1), rr, rr, signature=[A.signature[3], A.signature[4]])
    C2 = ST.from_array(np.eye(1), rr, rr, signature=[A.signature[4], A.signature[5]])
    C3 = ST.from_array(np.eye(1), rr, rr, signature=[A.signature[2], A.signature[5]])
    C4 = ST.from_array(np.eye(1), rr, rr, signature=[A.signature[2], A.signature[3]])

    rr = (ST.singlet(),)
    rc = (A.col_reps[2], A.col_reps[2], ST.singlet())
    sig = np.array([A.signature[3], A.signature[4], ~A.signature[4], A.signature[5]])
    T1 = ST.from_array(np.eye(A.shape[4])[None, :, :, None], rr, rc, signature=sig)

    rr = (ST.singlet(),)
    rc = (ST.singlet(), A.col_reps[3], A.col_reps[3])
    sig = np.array([A.signature[2], A.signature[4], A.signature[5], ~A.signature[5]])
    T2 = ST.from_array(np.eye(A.shape[5])[None, None, :, :], rr, rc, signature=sig)

    rr = (A.col_reps[0], A.col_reps[0], ST.singlet())
    rc = (ST.singlet(),)
    sig = np.array([A.signature[2], ~A.signature[2], A.signature[3], A.signature[5]])
    T3 = ST.from_array(np.eye(A.shape[2])[:, :, None, None], rr, rc, signature=sig)

    rr = (ST.singlet(),)
    rc = (A.col_reps[1], A.col_reps[1], ST.singlet())
    sig = np.array([A.signature[2], A.signature[3], ~A.signature[3], A.signature[4]])
    T4 = ST.from_array(np.eye(A.shape[3])[None, :, :, None], rr, rc, signature=sig)
    return C1, T1, C2, T2, C3, T3, C4, T4


class CTM_Environment:
    """
    Container for CTMRG environment tensors. Follow leg conventions from CTMRG.
    """

    def __init__(self, cell, As, C1s, C2s, C3s, C4s, T1s, T2s, T3s, T4s):
        """
        Initialize environment tensors. Consider calling from_elementary_tensors
        or from_file instead of directly calling this function.
        """
        # 1) Define indices and non-equivalent site_coordinates from cell
        # construct list of unique letters sorted according to appearance order in cell
        # (may be different from lexicographic order)
        seen = set()
        seen_add = seen.add
        letters = [c for c in cell.flat if not (c in seen or seen_add(c))]
        self._n_sites = len(letters)

        # [row,col] indices are transposed from (x,y) coordinates but (x,y) is natural
        # to specify positions, so we need to transpose indices here to get simple CTMRG
        # code. Construct indices and site_coords such that
        # -for all site i, i == indices[site_coords[i][0], site_coords[i][1]]
        # -for all site_coords (x,y), (x,y) == site_coords[indices[x,y]]
        self._site_coords = np.empty((self._n_sites, 2), dtype=np.int8)
        indices = np.empty(cell.shape, dtype=int)
        for i, c in enumerate(letters):
            inds_c = cell == c  # a tensor can appear more than once in tiling
            ind_values = inds_c.nonzero()
            self._site_coords[i] = ind_values[1][0], ind_values[0][0]  # transpose
            indices[inds_c] = i

        self._Ly, self._Lx = cell.shape
        self._cell = cell
        self._indices = indices.T.copy()  # transpose

        # 2) store tensors. They have to follow cell.flat order.
        self._neq_As = tuple(As)
        self._neq_C1s = C1s
        self._neq_C2s = C2s
        self._neq_C3s = C3s
        self._neq_C4s = C4s
        self._neq_T1s = T1s
        self._neq_T2s = T2s
        self._neq_T3s = T3s
        self._neq_T4s = T4s
        self._Dmin = min(min(A.shape[2:]) for A in self._neq_As)
        self._Dmax = max(max(A.shape[2:]) for A in self._neq_As)

        # 3) check tensor types and numbers
        ST = type(As[0])
        self._symmetry = ST.symmetry()

        def check(tensors, ndim, name):
            if len(tensors) != self._n_sites:
                raise ValueError(f"Invalid {name} number")
            for t in tensors:
                if type(t) is not ST:
                    raise ValueError(f"Invalid {name} type")
                if t.ndim != ndim:
                    raise ValueError(f"Invalid {name} ndim")

        check(As, 6, "A")
        check(C1s, 2, "C1")
        check(C2s, 2, "C2")
        check(C3s, 2, "C3")
        check(C4s, 2, "C4")
        check(T1s, 4, "T1")
        check(T2s, 4, "T2")
        check(T3s, 4, "T3")
        check(T4s, 4, "T4")

        # check matching between C, T and A requires cumbersome dealing with row/col
        # and conjugation. Little risk of error, _initialize_env and from_file are safe.

        # 4) check elementary tensors match together (need cell and coords)
        for x, y in self._site_coords:
            A = self.get_A(x, y)
            Ay = self.get_A(x, y - 1)
            if A.signature[2] == Ay.signature[4]:
                raise ValueError(f"signature mismatch between {(x, y)} and {(x, y-1)}")
            if (A.col_reps[0] != Ay.col_reps[2]).any():
                raise ValueError(f"rep mismatch between {(x, y)} and {(x, y-1)}")
            Ax = self.get_A(x + 1, y)
            if A.signature[3] == Ax.signature[5]:
                raise ValueError(f"signature mismatch between {(x+1, y)} and {(x, y)}")
            if (A.col_reps[1] != Ax.col_reps[3]).any():
                raise ValueError(f"rep mismatch between {(x+1, y)} and {(x, y)}")

        # 5) init enlarged corner and projectors lists
        self._corners_ul = [None] * self._n_sites
        self._corners_ur = [None] * self._n_sites
        self._corners_dl = [None] * self._n_sites
        self._corners_dr = [None] * self._n_sites
        self._reset_temp_lists()

    @property
    def Dmax(self):
        return self._Dmax

    @property
    def Dmin(self):
        return self._Dmin

    @property
    def Lx(self):
        return self._Lx

    @property
    def Ly(self):
        return self._Ly

    @property
    def cell(self):
        return self._cell

    @property
    def chi_values(self):
        s = set()
        for C in self._neq_C1s + self._neq_C3s:
            s.add(C.shape[0])
            s.add(C.shape[1])
        return sorted(s)

    def get_corner_representations(self):
        unique = []
        s = set()
        for C in self._neq_C1s + self._neq_C3s:
            r = C.row_reps[0]
            rt = tuple(r.ravel())
            if rt not in s:
                s.add(rt)
                unique.append(r)
            r = C.col_reps[0]
            rt = tuple(r.ravel())
            if rt not in s:
                s.add(rt)
                unique.append(r)

        def sortkey(r):
            return (C.representation_dimension(r), tuple(r.ravel()))

        return sorted(unique, key=sortkey)

    @property
    def elementary_tensors(self):
        return self._neq_As

    @property
    def n_sites(self):
        return self._n_sites

    @property
    def site_coords(self):
        return self._site_coords

    @classmethod
    def from_elementary_tensors(cls, tiling, tensors, dummy):
        """
        Construct CTM_Environment from elementary tensors according to tiling. If dummy,
        environment is initalized as dummy: corners are (1x1) identity matrices with
        trivial representation, edges are identiy matrices between layers with dummy
        legs for chi.

        Parameters
        ----------
        tiling : string
            Tiling pattern, such as 'AB\nBA' or 'AB\nCD'.
        tensors : iterable of n_sites SymmetricTensor
            Tensors given from left to right and up to down (as in array.flat)
        dummy : bool
            Whether to initalize the environment tensors as dummy or from site tensors.
        """
        txt = [" ".join(w) for w in tiling.strip().splitlines()]
        cell = np.atleast_2d(np.genfromtxt(txt, dtype="U1"))
        if len(tensors) != len(set(cell.flat)):
            raise ValueError("Incompatible cell and tensor number")
        C1s, C2s, C3s, C4s, T1s, T2s, T3s, T4s = ([] for i in range(8))
        for A in tensors:
            if dummy:
                C1, T1, C2, T2, C3, T3, C4, T4 = _initialize_dummy(A)
            else:
                C1, T1, C2, T2, C3, T3, C4, T4 = _initialize_env(A)
            C1s.append(C1)
            T1s.append(T1)
            C2s.append(C2)
            T2s.append(T2)
            C3s.append(C3)
            T3s.append(T3)
            C4s.append(C4)
            T4s.append(T4)
        return cls(cell, tensors, C1s, C2s, C3s, C4s, T1s, T2s, T3s, T4s)

    @classmethod
    def load_from_file(cls, savefile):
        """
        Construct CTM_Environment from save file.
        """
        As, C1s, C2s, C3s, C4s, T1s, T2s, T3s, T4s = ([] for i in range(9))
        with np.load(savefile) as data:
            cell = data["_CTM_cell"]
            ST = get_symmetric_tensor_type(data["_CTM_symmetry"][()])
            for i in range(len(set(cell.flat))):
                As.append(ST.load_from_dic(data, prefix=f"_CTM_A_{i}"))
                C1s.append(ST.load_from_dic(data, prefix=f"_CTM_C1_{i}"))
                C2s.append(ST.load_from_dic(data, prefix=f"_CTM_C2_{i}"))
                C3s.append(ST.load_from_dic(data, prefix=f"_CTM_C3_{i}"))
                C4s.append(ST.load_from_dic(data, prefix=f"_CTM_C4_{i}"))
                T1s.append(ST.load_from_dic(data, prefix=f"_CTM_T1_{i}"))
                T2s.append(ST.load_from_dic(data, prefix=f"_CTM_T2_{i}"))
                T3s.append(ST.load_from_dic(data, prefix=f"_CTM_T3_{i}"))
                T4s.append(ST.load_from_dic(data, prefix=f"_CTM_T4_{i}"))
        return cls(cell, As, C1s, C2s, C3s, C4s, T1s, T2s, T3s, T4s)

    # avoid property to mimic ST behavior
    def symmetry(self):
        return self._symmetry

    def get_data_dic(self):
        """
        Return environment data as a dict to save in external file.
        """
        data = {"_CTM_cell": self._cell, "_CTM_symmetry": self._symmetry}

        # use SymmetricTensor nice I/O, pure npz file without pickle
        for i in range(self._n_sites):
            data |= self._neq_As[i].get_data_dic(prefix=f"_CTM_A_{i}")
            data |= self._neq_C1s[i].get_data_dic(prefix=f"_CTM_C1_{i}")
            data |= self._neq_C2s[i].get_data_dic(prefix=f"_CTM_C2_{i}")
            data |= self._neq_C3s[i].get_data_dic(prefix=f"_CTM_C3_{i}")
            data |= self._neq_C4s[i].get_data_dic(prefix=f"_CTM_C4_{i}")
            data |= self._neq_T1s[i].get_data_dic(prefix=f"_CTM_T1_{i}")
            data |= self._neq_T2s[i].get_data_dic(prefix=f"_CTM_T2_{i}")
            data |= self._neq_T3s[i].get_data_dic(prefix=f"_CTM_T3_{i}")
            data |= self._neq_T4s[i].get_data_dic(prefix=f"_CTM_T4_{i}")
        return data

    def reset_constructed_corners(self):
        self._corners_ul = [None] * self._n_sites
        self._corners_ur = [None] * self._n_sites
        self._corners_dl = [None] * self._n_sites
        self._corners_dr = [None] * self._n_sites

    def set_tensors(self, tensors):
        """
        Set new elementary tensors. Type, shape and representations have to match
        current elementary tensors.
        """
        if len(tensors) != self._n_sites:
            raise ValueError("Incompatible cell and tensors")
        for i, A in enumerate(tensors):
            if not A.match_representations(self._neq_As[i]):
                raise ValueError("Incompatible representation for new tensor")

        self._neq_As = tuple(tensors)
        self.reset_constructed_corners()

    def set_symmetry(self, symmetry):
        if symmetry != self._symmetry:
            self.reset_constructed_corners()
            self._symmetry = symmetry

            self._neq_As = tuple(A.cast(symmetry) for A in self._neq_As)
            self._neq_C1s = [C1.cast(symmetry) for C1 in self._neq_C1s]
            self._neq_C2s = [C2.cast(symmetry) for C2 in self._neq_C2s]
            self._neq_C3s = [C3.cast(symmetry) for C3 in self._neq_C3s]
            self._neq_C4s = [C4.cast(symmetry) for C4 in self._neq_C4s]
            self._neq_T1s = [T1.cast(symmetry) for T1 in self._neq_T1s]
            self._neq_T2s = [T2.cast(symmetry) for T2 in self._neq_T2s]
            self._neq_T3s = [T3.cast(symmetry) for T3 in self._neq_T3s]
            self._neq_T4s = [T4.cast(symmetry) for T4 in self._neq_T4s]

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
        self._neq_P = [None] * self._n_sites
        self._neq_Pt = [None] * self._n_sites
        self._nCX = [None] * self._n_sites
        self._nT = [None] * self._n_sites
        self._nCY = [None] * self._n_sites

    def set_renormalized_tensors_up(self):
        self._neq_C1s = self._nCX
        self._neq_T1s = self._nT
        self._neq_C2s = self._nCY
        self._reset_temp_lists()

    def set_renormalized_tensors_right(self):
        self._neq_C2s = self._nCX
        self._neq_T2s = self._nT
        self._neq_C3s = self._nCY
        self._reset_temp_lists()

    def set_renormalized_tensors_down(self):
        self._neq_C3s = self._nCX
        self._neq_T3s = self._nT
        self._neq_C4s = self._nCY
        self._reset_temp_lists()

    def set_renormalized_tensors_left(self):
        self._neq_C4s = self._nCX
        self._neq_T4s = self._nT
        self._neq_C1s = self._nCY
        self._reset_temp_lists()