import numpy as np

from toolsU1 import default_color, combine_colors, BlockMatrixU1


def _initialize_env(A):
    #
    #   C1-0  3-T1-0  1-C2            0
    #   |       ||       |             \ 2
    #   1       12       0              \|
    #                                  5-A-3
    #   0       0        0               |\
    #   |       |        |               4 1
    #   T4=1  3-a--1  2=T2
    #   |  2    |     3  |
    #   3       2        1
    #
    #   0       01       0
    #   |       ||       |
    #   C4-1  3-T3-2  1-C3
    #
    a = (
        np.tensordot(A, A.conj(), ((0, 1), (0, 1)))
        .transpose(0, 4, 1, 5, 2, 6, 3, 7)
        .copy()
    )
    C1 = np.einsum("aacdefgg->cdef", a).reshape(A.shape[3] ** 2, A.shape[4] ** 2)
    T1 = np.einsum("aacdefgh->cdefgh", a).reshape(
        A.shape[3] ** 2, A.shape[4], A.shape[4], A.shape[5] ** 2
    )
    C2 = np.einsum("aaccefgh->efgh", a).reshape(A.shape[4] ** 2, A.shape[5] ** 2)
    T2 = np.einsum("abccefgh->abefgh", a).reshape(
        A.shape[2] ** 2, A.shape[4] ** 2, A.shape[5], A.shape[5]
    )
    C3 = np.einsum("abcceegh->abgh", a).reshape(A.shape[2] ** 2, A.shape[5] ** 2)
    T3 = np.einsum("abcdeegh->abcdgh", a).reshape(
        A.shape[2], A.shape[2], A.shape[3] ** 2, A.shape[5] ** 2
    )
    C4 = np.einsum("abcdeegg->abcd", a).reshape(A.shape[2] ** 2, A.shape[3] ** 2)
    T4 = np.einsum("abcdefgg->abcdef", a).reshape(
        A.shape[2] ** 2, A.shape[3], A.shape[3], A.shape[4] ** 2
    )
    return C1, T1, C2, T4, T2, C4, T3, C3


def _block_AAconj(A, col_A):
    """
    Construct BlockMatrixU1 versions of double layer tensor a = A-A* that can be used in
    add_a_blockU1. a is a matrix with merged bra and ket legs *and* legs merged in two
    directions as rows and as columns. One version for each corner is needed, so we have
    a_ul, a_ur, a_dl and a_dr. However to save memory, a_dl and a_dr can be defined as
    a_ur.T and a_dl. and use same memory storage.
    To be able to use a_ur and a_ul in the same function, unconventional leg order is
    required in a_ur. Here, we use 4-legs tensor to specify leg ordering, but as
    BlockMatrixU1 matrices legs 0 and 1 are merged, so are legs 2 and 3.
       2                       3
       ||                      ||
    3=a_ul=0                1=a_ur=2
       ||                      ||
        1                       0

       1                       0
       ||                      ||
    3=a_dl=0 = a_ur.T       1=a_dr=2 = a_ul.T
       ||                      ||
        2                       3
    """
    a = np.tensordot(A, A.conj(), ((0, 1), (0, 1)))
    col_u = combine_colors(col_A[2], -col_A[2])
    col_r = combine_colors(col_A[3], -col_A[3])
    col_d = combine_colors(col_A[4], -col_A[4])
    col_l = combine_colors(col_A[5], -col_A[5])
    c_ul = combine_colors(col_u, col_l)
    c_ur = combine_colors(col_r, col_u)  # swap u and r to use ul function
    c_rd = combine_colors(col_r, col_d)
    c_dl = combine_colors(col_d, col_l)
    # a_ul used to contract corner_ul: u and l legs are *last* for a_ul @ TT
    a_ul = a.transpose(1, 5, 2, 6, 0, 4, 3, 7).reshape(c_rd.size, c_ul.size)
    a_ul = BlockMatrixU1.from_dense(a_ul, -c_rd, c_ul)
    a_ur = a.transpose(2, 6, 3, 7, 1, 5, 0, 4).reshape(c_dl.size, c_ur.size)
    a_ur = BlockMatrixU1.from_dense(a_ur, -c_dl, c_ur)
    return a_ul, a_ur, c_ul, c_ur, c_rd, c_dl


def _color_correspondence(old_col, new_col):
    """
    Find correspondances between old and new set of colors on a given axis. Return same
    size arrays with same color structure.
    """
    old_rows = []
    new_rows = []
    for c in set(new_col):
        oldrc = (old_col == c).nonzero()[0]
        newrc = (new_col == c).nonzero()[0]
        old_rows += list(oldrc[: len(newrc)])
        new_rows += list(newrc[: len(oldrc)])
    old_rows = np.array(old_rows)
    new_rows = np.array(new_rows)
    s = old_rows.argsort(kind="stable")  # minimal disruption
    old_rows = old_rows[s]  # != range(d) if some rows are removed
    new_rows = new_rows[s]  # this is a bit tedious, but optimises copy
    return old_rows, new_rows


class CTM_Environment(object):
    """
    Container for CTMRG environment tensors. Follow leg conventions from CTMRG.
    """

    def __init__(
        self,
        cell,
        neq_As,
        neq_C1s,
        neq_T1s,
        neq_C2s,
        neq_T2s,
        neq_C3s,
        neq_T3s,
        neq_C4s,
        neq_T4s,
        colors_A=None,
        colors_C1_r=None,
        colors_C1_d=None,
        colors_C2_d=None,
        colors_C2_l=None,
        colors_C3_u=None,
        colors_C3_l=None,
        colors_C4_u=None,
        colors_C4_r=None,
    ):
        """
        Initialize environment tensors. Consider calling from_elementary_tensors
        or from_file instead of directly calling this function.
        """

        # 1 Define indices and neq_coords from cell
        # construct list of unique letters sorted according to appearance order in cell
        # (may be different from lexicographic order)
        seen = set()
        seen_add = seen.add
        letters = [c for c in cell.flat if not (c in seen or seen_add(c))]

        self._Nneq = len(letters)
        assert (
            self._Nneq
            == len(neq_As)
            == len(neq_C1s)
            == len(neq_T1s)
            == len(neq_C2s)
            == len(neq_T2s)
            == len(neq_C3s)
            == len(neq_T3s)
            == len(neq_C4s)
            == len(neq_T4s)
        )

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
        self._neq_C1s = neq_C1s
        self._neq_T1s = neq_T1s
        self._neq_C2s = neq_C2s
        self._neq_T2s = neq_T2s
        self._neq_C3s = neq_C3s
        self._neq_T3s = neq_T3s
        self._neq_C4s = neq_C4s
        self._neq_T4s = neq_T4s

        # 3) initialize temp arrays
        self._corners_ul = [None] * self._Nneq
        self._corners_ur = [None] * self._Nneq
        self._corners_dl = [None] * self._Nneq
        self._corners_dr = [None] * self._Nneq
        self._reset_temp_lists()

        # 4) if colors are provided, store them, else define them as default_colors
        if colors_A is not None:
            assert (
                self._Nneq
                == len(colors_A)
                == len(colors_C1_r)
                == len(colors_C1_d)
                == len(colors_C2_d)
                == len(colors_C2_l)
                == len(colors_C3_u)
                == len(colors_C3_l)
                == len(colors_C4_u)
                == len(colors_C4_r)
            )
            self._colors_A = colors_A
            self._colors_C1_r = colors_C1_r
            self._colors_C1_d = colors_C1_d
            self._colors_C2_d = colors_C2_d
            self._colors_C2_l = colors_C2_l
            self._colors_C3_u = colors_C3_u
            self._colors_C3_l = colors_C3_l
            self._colors_C4_u = colors_C4_u
            self._colors_C4_r = colors_C4_r

            # store blockwise A*A* to construct corners + colors (cannot tranpose lists)
            self._a_col_ur = []
            self._a_col_ul = []
            self._a_col_rd = []
            self._a_col_dl = []
            for A, col_A in zip(neq_As, colors_A):
                a_ul, a_ur, col_ul, col_ur, col_rd, col_dl = _block_AAconj(A, col_A)
                self._a_col_ur.append((a_ur, col_dl, col_ur))
                self._a_col_ul.append((a_ul, col_rd, col_ul))
                self._a_col_rd.append((a_ul.T, col_ul, col_rd))
                self._a_col_dl.append((a_ur.T, col_ur, col_dl))
        else:
            self._colors_A = [(default_color,) * 6] * self._Nneq
            self._colors_C1_r = [default_color] * self._Nneq
            self._colors_C1_d = [default_color] * self._Nneq
            self._colors_C2_d = [default_color] * self._Nneq
            self._colors_C2_l = [default_color] * self._Nneq
            self._colors_C3_u = [default_color] * self._Nneq
            self._colors_C3_l = [default_color] * self._Nneq
            self._colors_C4_u = [default_color] * self._Nneq
            self._colors_C4_r = [default_color] * self._Nneq

    def restart(self):
        """
        Erase current environment tensors C and T and restart them from elementary
        tensors.
        """
        for i, A in enumerate(self._neq_As):
            C1, T1, C2, T4, T2, C4, T3, C3 = _initialize_env(A)
            self._neq_C1s[i] = C1
            self._neq_T1s[i] = T1
            self._neq_C2s[i] = C2
            self._neq_T2s[i] = T2
            self._neq_C3s[i] = C3
            self._neq_T3s[i] = T3
            self._neq_C4s[i] = C4
            self._neq_T4s[i] = T4

        if self._colors_A[0][0].size:
            for i, colA in enumerate(self._colors_A):
                c2 = combine_colors(colA[2], -colA[2])
                c3 = combine_colors(colA[3], -colA[3])
                c4 = combine_colors(colA[4], -colA[4])
                c5 = combine_colors(colA[5], -colA[5])
                self._colors_C1_r[i] = c3
                self._colors_C1_d[i] = c4
                self._colors_C2_d[i] = c4
                self._colors_C2_l[i] = c5
                self._colors_C3_u[i] = c2
                self._colors_C3_l[i] = c5
                self._colors_C4_u[i] = c2
                self._colors_C4_r[i] = c3

        self._corners_ul = [None] * self._Nneq
        self._corners_ur = [None] * self._Nneq
        self._corners_dl = [None] * self._Nneq
        self._corners_dr = [None] * self._Nneq
        self._reset_temp_lists()

    @classmethod
    def from_elementary_tensors(cls, tensors, tiling, colors=None):
        """
        Construct CTM_Environment from elementary tensors according to tiling.

        Parameters
        ----------
        tensors : iterable of Nneq numpy arrays
          Tensors given from left to right and up to down (as in array.flat)
        tiling : string
          Tiling pattern, such as 'AB\nCD'.
        colors : optional, list of Nneq colors
          U(1) quantum numbers corresponding to the tensors. Note that dimensions are
          check for compatibility with tensors, but color compatibility between legs to
          contract is not checked.
        """
        cell = np.genfromtxt(
            [" ".join(w) for w in tiling.strip().splitlines()], dtype="U1"
        )
        Nneq = len(set(cell.flat))
        if len(tensors) != Nneq:
            raise ValueError("Tensor number do not match tiling")

        # store tensors according to cell.flat order
        neq_As = []
        neq_C1s = []
        neq_T1s = []
        neq_C2s = []
        neq_T2s = []
        neq_C3s = []
        neq_T3s = []
        neq_C4s = []
        neq_T4s = []

        for A0 in tensors:
            A = np.ascontiguousarray(A0)
            if A.ndim == 5:  # if no ancilla, add 1
                A = A.reshape(
                    A.shape[0], 1, A.shape[1], A.shape[2], A.shape[3], A.shape[4]
                )
            if A.ndim != 6:
                raise ValueError("Elementary tensor must be of rank 5 or 6")
            C1, T1, C2, T4, T2, C4, T3, C3 = _initialize_env(A)
            neq_As.append(A)
            neq_C1s.append(C1)
            neq_T1s.append(T1)
            neq_C2s.append(C2)
            neq_T2s.append(T2)
            neq_C3s.append(C3)
            neq_T3s.append(T3)
            neq_C4s.append(C4)
            neq_T4s.append(T4)

        if colors is None or sum(sum(c.size for c in t) for t in colors) == 0:
            return cls(  # sum(sum deals with default_color as input
                cell,
                neq_As,
                neq_C1s,
                neq_T1s,
                neq_C2s,
                neq_T2s,
                neq_C3s,
                neq_T3s,
                neq_C4s,
                neq_T4s,
            )
        if len(colors) != Nneq:
            raise ValueError("Color number do not match tensors")
        colors_A = []
        # more convenient to store separetly row and column colors of corners
        colors_C1_r = []
        colors_C1_d = []
        colors_C2_d = []
        colors_C2_l = []
        colors_C3_u = []
        colors_C3_l = []
        colors_C4_u = []
        colors_C4_r = []
        for A, colA in zip(tensors, colors):
            if tuple(len(c) for c in colA) != A.shape:
                raise ValueError("Colors do not match tensors")
            if len(colA) == 5:  # add empty ancilla
                colA = (colA[0], np.zeros(1, dtype=np.int8), *colA[1:])
            c2 = combine_colors(colA[2], -colA[2])
            c3 = combine_colors(colA[3], -colA[3])
            c4 = combine_colors(colA[4], -colA[4])
            c5 = combine_colors(colA[5], -colA[5])
            colors_A.append(colA)
            colors_C1_r.append(c3)
            colors_C1_d.append(c4)
            colors_C2_d.append(c4)
            colors_C2_l.append(c5)
            colors_C3_u.append(c2)
            colors_C3_l.append(c5)
            colors_C4_u.append(c2)
            colors_C4_r.append(c3)
        return cls(
            cell,
            neq_As,
            neq_C1s,
            neq_T1s,
            neq_C2s,
            neq_T2s,
            neq_C3s,
            neq_T3s,
            neq_C4s,
            neq_T4s,
            colors_A,
            colors_C1_r,
            colors_C1_d,
            colors_C2_d,
            colors_C2_l,
            colors_C3_u,
            colors_C3_l,
            colors_C4_u,
            colors_C4_r,
        )

    def save_to_file(self, filename, chi, additional_data={}):
        """
        Save all tensors into external .npz file.

        Parameters
        ----------
        filename: str
          Name of the storage file.
        additional_data: dict
          Data to store together with environment data. Keys have to be string type.
        """
        # do not store lists to avoid pickle
        # come back to elementary numpy arrays
        data = {"_CTM_chi": chi, "_CTM_cell": self._cell}

        for i in range(self._Nneq):
            data[f"_CTM_A_{i}"] = self._neq_As[i]
            data[f"_CTM_C1_{i}"] = self._neq_C1s[i]
            data[f"_CTM_T1_{i}"] = self._neq_T1s[i]
            data[f"_CTM_C2_{i}"] = self._neq_C2s[i]
            data[f"_CTM_T2_{i}"] = self._neq_T2s[i]
            data[f"_CTM_C3_{i}"] = self._neq_C3s[i]
            data[f"_CTM_T3_{i}"] = self._neq_T3s[i]
            data[f"_CTM_C4_{i}"] = self._neq_C4s[i]
            data[f"_CTM_T4_{i}"] = self._neq_T4s[i]
            data[f"_CTM_colors_C1_r_{i}"] = self._colors_C1_r[i]
            data[f"_CTM_colors_C1_d_{i}"] = self._colors_C1_d[i]
            data[f"_CTM_colors_C2_d_{i}"] = self._colors_C2_d[i]
            data[f"_CTM_colors_C2_l_{i}"] = self._colors_C2_l[i]
            data[f"_CTM_colors_C3_u_{i}"] = self._colors_C3_u[i]
            data[f"_CTM_colors_C3_l_{i}"] = self._colors_C3_l[i]
            data[f"_CTM_colors_C4_u_{i}"] = self._colors_C4_u[i]
            data[f"_CTM_colors_C4_r_{i}"] = self._colors_C4_r[i]
            for leg in range(6):
                data[f"_CTM_colors_A_{i}_{leg}"] = self._colors_A[i][leg]

        np.savez_compressed(filename, **data, **additional_data)

    @classmethod
    def from_file(cls, filename):
        """
        Construct CTM_Environment from save file.
        """
        with np.load(filename) as data:
            chi = data["_CTM_chi"]
            cell = data["_CTM_cell"]
            Nneq = len(set(cell.flat))

            neq_As = [None] * Nneq
            neq_C1s = [None] * Nneq
            neq_T1s = [None] * Nneq
            neq_C2s = [None] * Nneq
            neq_T2s = [None] * Nneq
            neq_C3s = [None] * Nneq
            neq_T3s = [None] * Nneq
            neq_C4s = [None] * Nneq
            neq_T4s = [None] * Nneq

            # colors are always defined and stored, even if they are default_color
            colors_A = [None] * Nneq
            colors_C1_r = [None] * Nneq
            colors_C1_d = [None] * Nneq
            colors_C2_d = [None] * Nneq
            colors_C2_l = [None] * Nneq
            colors_C3_u = [None] * Nneq
            colors_C3_l = [None] * Nneq
            colors_C4_u = [None] * Nneq
            colors_C4_r = [None] * Nneq

            for i in range(Nneq):
                neq_As[i] = data[f"_CTM_A_{i}"]
                neq_C1s[i] = data[f"_CTM_C1_{i}"]
                neq_T1s[i] = data[f"_CTM_T1_{i}"]
                neq_C2s[i] = data[f"_CTM_C2_{i}"]
                neq_T2s[i] = data[f"_CTM_T2_{i}"]
                neq_C3s[i] = data[f"_CTM_C3_{i}"]
                neq_T3s[i] = data[f"_CTM_T3_{i}"]
                neq_C4s[i] = data[f"_CTM_C4_{i}"]
                neq_T4s[i] = data[f"_CTM_T4_{i}"]
                colors_A[i] = tuple(
                    data[f"_CTM_colors_A_{i}_{leg}"] for leg in range(6)
                )
                colors_C1_r[i] = data[f"_CTM_colors_C1_r_{i}"]
                colors_C1_d[i] = data[f"_CTM_colors_C1_d_{i}"]
                colors_C2_d[i] = data[f"_CTM_colors_C2_d_{i}"]
                colors_C2_l[i] = data[f"_CTM_colors_C2_l_{i}"]
                colors_C3_u[i] = data[f"_CTM_colors_C3_u_{i}"]
                colors_C3_l[i] = data[f"_CTM_colors_C3_l_{i}"]
                colors_C4_u[i] = data[f"_CTM_colors_C4_u_{i}"]
                colors_C4_r[i] = data[f"_CTM_colors_C4_r_{i}"]

        return (
            chi,
            cls(
                cell,
                neq_As,
                neq_C1s,
                neq_T1s,
                neq_C2s,
                neq_T2s,
                neq_C3s,
                neq_T3s,
                neq_C4s,
                neq_T4s,
                colors_A,
                colors_C1_r,
                colors_C1_d,
                colors_C2_d,
                colors_C2_l,
                colors_C3_u,
                colors_C3_l,
                colors_C4_u,
                colors_C4_r,
            ),
        )

    def set_tensors(self, tensors, colors=None):
        if self._Nneq != len(tensors):
            raise ValueError("Incompatible cell and tensors")
        if colors is not None and sum(sum(c.size for c in t) for t in colors):
            colorized = True
            if self._Nneq != len(colors):
                raise ValueError("Incompatible cell and colors")
        else:
            colorized = False
        for i, A in enumerate(tensors):  # or self._neq_coords?
            A = np.ascontiguousarray(A)
            if A.ndim == 5:  # if no ancilla, add 1
                A = A.reshape(
                    A.shape[0], 1, A.shape[1], A.shape[2], A.shape[3], A.shape[4]
                )
            if A.ndim != 6:
                raise ValueError("Elementary tensor must be of rank 5 or 6")
            oldA = self._neq_As[i]
            oldcol = self._colors_A[i]
            if colorized:
                col = colors[i]
                if len(col) == 5:
                    col = (col[0], np.zeros(1, dtype=np.int8), *col[1:])
                if tuple(len(c) for c in col) != A.shape:
                    raise ValueError("Colors do not match tensors")
                a_ul, a_ur, col_ul, col_ur, col_rd, col_dl = _block_AAconj(A, col)
                self._a_col_ur[i] = (a_ur, col_dl, col_ur)
                self._a_col_ul[i] = (a_ul, col_rd, col_ul)
                self._a_col_rd[i] = (a_ul.T, col_ul, col_rd)
                self._a_col_dl[i] = (a_ur.T, col_ur, col_dl)
            else:
                col = (default_color,) * 6
            if (
                A.shape[0] != oldA.shape[0]
                or A.shape[1] != oldA.shape[1]
                or (col[0] != oldcol[0]).any()
                or (col[1] != oldcol[1]).any()
            ):
                # not a problem for the code, but physically meaningless
                raise ValueError(
                    "Physical and ancilla dimensions and colors cannot change"
                )

            # when the shape of a tensor (or its quantum numbers) changes, we still
            # would like to keep the environment. This is still possible: A legs were
            # obtained from some an SVD then truncated, ie some singular value was put
            # to 0 (either in former or current A). Just add a row of zero corresponding
            # to this 0 singular value in the tensor that misses it and dimensions
            # match. In case of U(1) symmetry, this is the same but sector wise.

            x, y = self._neq_coords[i]
            # up axis
            if A.shape[2] != oldA.shape[2] or (oldcol[2] != col[2]).any():
                j = self._indices[x % self._Lx, (y - 1) % self._Ly]
                oldT1 = self._neq_T1s[j]
                newT1 = np.zeros(
                    (oldT1.shape[0], A.shape[2], A.shape[2], oldT1.shape[3])
                )
                if oldcol[2].size:  # colorwise copy
                    old_rows, new_rows = _color_correspondence(
                        oldcol[2], col[2]
                    )  # put copy outside of color loop
                else:  # colors are not provided
                    old_rows = slice(0, oldA.shape[2])
                    new_rows = slice(0, A.shape[2])
                newT1[:, new_rows, new_rows] = oldT1[:, old_rows, old_rows]
                self._neq_T1s[j] = newT1

            # right axis
            if A.shape[3] != oldA.shape[3] or (oldcol[3] != col[3]).any():
                j = self._indices[(x + 1) % self._Lx, y % self._Ly]
                oldT2 = self._neq_T2s[j]
                newT2 = np.zeros(
                    (oldT2.shape[0], oldT2.shape[1], A.shape[3], A.shape[3])
                )
                if oldcol[3].size:  # colorwise copy
                    old_rows, new_rows = _color_correspondence(oldcol[3], col[3])
                else:  # colors are not provided
                    old_rows = slice(0, oldA.shape[3])
                    new_rows = slice(0, A.shape[3])
                newT2[:, :, new_rows, new_rows] = oldT2[:, :, old_rows, old_rows]
                self._neq_T2s[j] = newT2

            if A.shape[4] != oldA.shape[4] or (oldcol[4] != col[4]).any():
                j = self._indices[x % self._Lx, (y + 1) % self._Ly]
                oldT3 = self._neq_T3s[j]
                newT3 = np.zeros(
                    (A.shape[4], A.shape[4], oldT3.shape[2], oldT3.shape[3])
                )
                if oldcol[4].size:  # colorwise copy
                    old_rows, new_rows = _color_correspondence(oldcol[4], col[4])
                else:  # colors are not provided
                    old_rows = slice(0, oldA.shape[4])
                    new_rows = slice(0, A.shape[4])
                newT3[new_rows, new_rows] = oldT3[old_rows, old_rows]
                self._neq_T3s[j] = newT3

            if A.shape[5] != oldA.shape[5] or (oldcol[5] != col[5]).any():
                j = self._indices[(x - 1) % self._Lx, y % self._Ly]
                oldT4 = self._neq_T4s[j]
                newT4 = np.zeros(
                    (oldT4.shape[0], A.shape[5], A.shape[5], oldT4.shape[3])
                )
                if oldcol[5].size:  # colorwise copy
                    old_rows, new_rows = _color_correspondence(oldcol[5], col[5])
                else:  # colors are not provided
                    old_rows = slice(0, oldA.shape[5])
                    new_rows = slice(0, A.shape[5])
                newT4[:, new_rows, new_rows] = oldT4[:, old_rows, old_rows]
                self._neq_T4s[j] = newT4

            self._neq_As[i] = A
            self._colors_A[i] = col

        # reset all corners C-T // T-A since A changed
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

    def get_colors_A(self, x, y):
        return self._colors_A[self._indices[x % self._Lx, y % self._Ly]]

    def get_C1(self, x, y):
        return self._neq_C1s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C1_r(self, x, y):
        return self._colors_C1_r[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C1_d(self, x, y):
        return self._colors_C1_d[self._indices[x % self._Lx, y % self._Ly]]

    def get_T1(self, x, y):
        return self._neq_T1s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_T1_r(self, x, y):
        return -self._colors_C2_l[self._indices[(x + 1) % self._Lx, y % self._Ly]]

    def get_color_T1_d(self, x, y):
        return -self._colors_A[self._indices[x % self._Lx, (y + 1) % self._Ly]][2]

    def get_color_T1_l(self, x, y):
        return -self._colors_C1_r[self._indices[(x - 1) % self._Lx, y % self._Ly]]

    def get_C2(self, x, y):
        return self._neq_C2s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C2_d(self, x, y):
        return self._colors_C2_d[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C2_l(self, x, y):
        return self._colors_C2_l[self._indices[x % self._Lx, y % self._Ly]]

    def get_T2(self, x, y):
        return self._neq_T2s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_T2_u(self, x, y):
        return -self._colors_C2_d[self._indices[x % self._Lx, (y - 1) % self._Ly]]

    def get_color_T2_d(self, x, y):
        return -self._colors_C3_u[self._indices[x % self._Lx, (y + 1) % self._Ly]]

    def get_color_T2_l(self, x, y):
        return -self._colors_A[self._indices[(x - 1) % self._Lx, y % self._Ly]][3]

    def get_C3(self, x, y):
        return self._neq_C3s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C3_u(self, x, y):
        return self._colors_C3_u[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C3_l(self, x, y):
        return self._colors_C3_l[self._indices[x % self._Lx, y % self._Ly]]

    def get_T3(self, x, y):
        return self._neq_T3s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_T3_u(self, x, y):
        return -self._colors_A[self._indices[x % self._Lx, (y - 1) % self._Ly]][4]

    def get_color_T3_r(self, x, y):
        return -self._colors_C3_l[self._indices[(x + 1) % self._Lx, y % self._Ly]]

    def get_color_T3_l(self, x, y):
        return -self._colors_C4_r[self._indices[(x - 1) % self._Lx, y % self._Ly]]

    def get_C4(self, x, y):
        return self._neq_C4s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C4_u(self, x, y):
        return self._colors_C4_u[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_C4_r(self, x, y):
        return self._colors_C4_r[self._indices[x % self._Lx, y % self._Ly]]

    def get_T4(self, x, y):
        return self._neq_T4s[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_T4_u(self, x, y):
        return -self._colors_C1_d[self._indices[x % self._Lx, (y - 1) % self._Ly]]

    def get_color_T4_r(self, x, y):
        return -self._colors_A[self._indices[(x + 1) % self._Lx, y % self._Ly]][5]

    def get_color_T4_d(self, x, y):
        return -self._colors_C4_u[self._indices[x % self._Lx, (y + 1) % self._Ly]]

    def get_P(self, x, y):
        return self._neq_P[self._indices[x % self._Lx, y % self._Ly]]

    def get_color_P(self, x, y):
        return self._colors_P[self._indices[x % self._Lx, y % self._Ly]]

    def get_Pt(self, x, y):
        return self._neq_Pt[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_col_ul(self, x, y):
        return self._a_col_ul[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_col_ur(self, x, y):
        return self._a_col_ur[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_col_rd(self, x, y):
        return self._a_col_rd[self._indices[x % self._Lx, y % self._Ly]]

    def get_a_col_dl(self, x, y):
        return self._a_col_dl[self._indices[x % self._Lx, y % self._Ly]]

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

    def store_projectors(self, x, y, P, Pt, color_P=default_color):
        j = self._indices[x % self._Lx, y % self._Ly]
        self._neq_P[j] = P
        self._neq_Pt[j] = Pt
        self._colors_P[j] = color_P

    def store_renormalized_tensors(
        self, x, y, nCX, nT, nCY, color_P=default_color, color_Pt=default_color
    ):
        j = self._indices[x % self._Lx, y % self._Ly]
        self._nCX[j] = nCX
        self._nT[j] = nT
        self._nCY[j] = nCY
        self._colors_CX[j] = color_P
        self._colors_CY[j] = color_Pt

    def _reset_temp_lists(self):
        # reset is needed because no list copy occurs
        self._neq_P = [None] * self._Nneq
        self._neq_Pt = [None] * self._Nneq
        self._nCX = [None] * self._Nneq
        self._nT = [None] * self._Nneq
        self._nCY = [None] * self._Nneq
        self._colors_P = [None] * self._Nneq
        self._colors_CX = [None] * self._Nneq
        self._colors_CY = [None] * self._Nneq

    def fix_renormalized_up(self):
        self._neq_C1s = self._nCX
        self._neq_T1s = self._nT
        self._neq_C2s = self._nCY
        self._colors_C1_r = self._colors_CX
        self._colors_C2_l = self._colors_CY
        self._corners_ul = [None] * self._Nneq
        self._corners_ur = [None] * self._Nneq
        self._reset_temp_lists()

    def fix_renormalized_right(self):
        self._neq_C2s = self._nCX
        self._neq_T2s = self._nT
        self._neq_C3s = self._nCY
        self._colors_C2_d = self._colors_CX
        self._colors_C3_u = self._colors_CY
        self._corners_ur = [None] * self._Nneq
        self._corners_dr = [None] * self._Nneq
        self._reset_temp_lists()

    def fix_renormalized_down(self):
        self._neq_C3s = self._nCX
        self._neq_T3s = self._nT
        self._neq_C4s = self._nCY
        self._colors_C3_l = self._colors_CX
        self._colors_C4_r = self._colors_CY
        self._corners_dr = [None] * self._Nneq
        self._corners_dl = [None] * self._Nneq
        self._reset_temp_lists()

    def fix_renormalized_left(self):
        self._neq_C4s = self._nCX
        self._neq_T4s = self._nT
        self._neq_C1s = self._nCY
        self._colors_C4_u = self._colors_CX
        self._colors_C1_d = self._colors_CY
        self._corners_dl = [None] * self._Nneq
        self._corners_ul = [None] * self._Nneq
        self._reset_temp_lists()
