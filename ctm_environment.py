import numpy as np
from toolsU1 import default_color, combine_colors

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
  a = np.tensordot(A,A.conj(),((0,1),(0,1))).transpose(0,4,1,5,2,6,3,7).copy()
  C1 = np.einsum('aacdefgg->cdef', a).reshape(A.shape[3]**2,A.shape[4]**2)
  T1 = np.einsum('aacdefgh->cdefgh', a).reshape(A.shape[3]**2,A.shape[4],A.shape[4],A.shape[5]**2)
  C2 = np.einsum('aaccefgh->efgh', a).reshape(A.shape[4]**2,A.shape[5]**2)
  T2 = np.einsum('abccefgh->abefgh', a).reshape(A.shape[2]**2,A.shape[4]**2,A.shape[5],A.shape[5])
  C3 = np.einsum('abcceegh->abgh', a).reshape(A.shape[2]**2,A.shape[5]**2)
  T3 = np.einsum('abcdeegh->abcdgh', a).reshape(A.shape[2],A.shape[2],A.shape[3]**2,A.shape[5]**2)
  C4 = np.einsum('abcdeegg->abcd', a).reshape(A.shape[2]**2,A.shape[3]**2)
  T4 = np.einsum('abcdefgg->abcdef', a).reshape(A.shape[2]**2,A.shape[3],A.shape[3],A.shape[4]**2)
  return C1,T1,C2,T4,T2,C4,T3,C3


def _color_correspondence(old_col, new_col):
  """
  Find correspondances between old and new set of colors on a given axis.
  Return same size arrays with same color structure
  """
  old_rows = []
  new_rows = []
  for c in set(new_col):
    oldrc = (old_col==c).nonzero()[0]
    newrc = (new_col==c).nonzero()[0]
    old_rows += list(oldrc[:len(newrc)])
    new_rows += list(newrc[:len(oldrc)])
  old_rows = np.array(old_rows)
  new_rows = np.array(new_rows)
  s = old_rows.argsort()
  old_rows = old_rows[s]  # != range(d) if some rows are removed
  new_rows = new_rows[s]  # this is a bit tedious, but optimises copy
  return old_rows, new_rows


class CTM_Environment(object):
  """
  Container for CTMRG environment tensors.
  leg conventions:

     C1-T1-T1-C2
     |  |  |   |
     T4-a--a--T2
     |  |  |   |
     T4-a--a--T2
     |  |  |   |
     C4-T3-T3-C3
  """

  def __init__(self, tensors=(), cell=None, tiling=None, colors=None, saveFile=None):
    """
    Store tensors and return them according to a given tiling. Tiling can be
    provided as a string such as 'AB\nCD' or directly as a numpy array of char.

    Parameters
    ----------
    tensors: iterable of Nneq numpy arrays
      Tensors given from left to right and up to down (as in array.flat)
    cell: array of characters
      Elementary cell, each letter representing non-equivalent tensors.
    tiling: string, optional.
      Tiling pattern. If cell is not provided, parse tiling to construct it.
    colors: list of Nneq colors
      U(1) quantum numbers corresponding to the tensors. Note that dimensions
      are check for compatibility with tensors, but color compatibility between
      legs to contract is not checked.
    saveFile: string
      Restart environment from npz save. If present, other arguments are not
      read.
    """
    if saveFile is not None:
      self.load_from_file(saveFile)
      return

    if cell is None:
      if tiling is None:     # default constructor, then call load_from_file
        cell = np.array([],dtype='U1').reshape(0,0)  # need ndim=2 for Lx,Ly
      else:
        cell = np.genfromtxt([" ".join(w) for w in tiling.strip().splitlines()], dtype='U1')
    else:
      cell = np.asarray(cell)

    # construct list of unique letters sorted according to appearance order in
    # cell (may be different from lexicographic order)
    seen = set()
    seen_add = seen.add
    letters = [l for l in cell.flat if not (l in seen or seen_add(l))]

    self._Nneq = len(letters)
    if self._Nneq != len(tensors):
      raise ValueError("Incompatible cell and tensors")

    # [row,col] indices are transposed from (x,y) coordinates
    # but (x,y) is natural to specify positions
    # so we need to transpose indices here to get simple CTMRG code
    # construct indices and neq_coords such that
    # - for all i in range(Nneq), i == indices[neq_coords[i][0], neq_coords[i][1]]
    # - for all (x,y) in neq_coords, (x,y) == neq_coords[indices[x,y]]
    self._neq_coords = np.empty((self._Nneq,2), dtype=np.int8)
    indices = np.empty(cell.shape, dtype=int)
    for i,l in enumerate(letters):
      inds_l = cell == l    # a tensor can appear more than once in tiling
      ind_values = inds_l.nonzero()
      self._neq_coords[i] = ind_values[1][0], ind_values[0][0] # transpose
      indices[inds_l] = i

    self._Ly, self._Lx = cell.shape
    self._cell = cell
    self._indices = indices.T.copy()  # transpose

    # store tensors according to cell.flat order
    self._neq_As = []
    self._neq_C1s = []
    self._neq_T1s = []
    self._neq_C2s = []
    self._neq_T2s = []
    self._neq_C3s = []
    self._neq_T3s = []
    self._neq_C4s = []
    self._neq_T4s = []

    for A in tensors:
      A = np.ascontiguousarray(A)
      if A.ndim == 5:  # if no ancila, add 1
        A = A.reshape(A.shape[0],1,A.shape[1],A.shape[2],A.shape[3],A.shape[4])
      C1,T1,C2,T4,T2,C4,T3,C3 = _initialize_env(A)
      self._neq_As.append(A)
      self._neq_C1s.append(C1)
      self._neq_T1s.append(T1)
      self._neq_C2s.append(C2)
      self._neq_T2s.append(T2)
      self._neq_C3s.append(C3)
      self._neq_T3s.append(T3)
      self._neq_C4s.append(C4)
      self._neq_T4s.append(T4)

    if colors is not None:
      if len(colors) != self._Nneq:
        raise ValueError("Color number do not match tensors")
      self._colors_A = []
      # more convenient to store separetly row and column colors of corners
      self._colors_C1_r = []
      self._colors_C1_d = []
      self._colors_C2_d = []
      self._colors_C2_l = []
      self._colors_C3_u = []
      self._colors_C3_l = []
      self._colors_C4_u = []
      self._colors_C4_r = []
      for A,colA in zip(tensors,colors):
        if tuple(len(c) for c in colA) != A.shape:
          raise ValueError("Colors do not match tensors")
        if len(colA) == 5:  # add empty ancila
          colA = (colA[0],np.zeros(1,dtype=np.int8),*colA[1:])
        c2 = combine_colors(colA[2],-colA[2])
        c3 = combine_colors(colA[3],-colA[3])
        c4 = combine_colors(colA[4],-colA[4])
        c5 = combine_colors(colA[5],-colA[5])
        self._colors_A.append(colA)
        self._colors_C1_r.append(c3)
        self._colors_C1_d.append(c4)
        self._colors_C2_d.append(c4)
        self._colors_C2_l.append(c5)
        self._colors_C3_u.append(c2)
        self._colors_C3_l.append(c5)
        self._colors_C4_u.append(c2)
        self._colors_C4_r.append(c3)
    else:
      self._colors_A = [(default_color,)*6]*self._Nneq
      self._colors_C1_r = [default_color]*self._Nneq
      self._colors_C1_d = [default_color]*self._Nneq
      self._colors_C2_d = [default_color]*self._Nneq
      self._colors_C2_l = [default_color]*self._Nneq
      self._colors_C3_u = [default_color]*self._Nneq
      self._colors_C3_l = [default_color]*self._Nneq
      self._colors_C4_u = [default_color]*self._Nneq
      self._colors_C4_r = [default_color]*self._Nneq

    self._reset_projectors_temp()


  def save_to_file(self, saveFile):
    """
    Save all tensors into external .npz file
    """
    # do not store lists to avoid pickle
    # come back to elementary numpy arrays
    data = {}
    data["cell"] = self._cell
    data["indices"] = self._indices
    data['neq_coords'] = self._neq_coords

    for i in range(self._Nneq):
      data[f"A_{i}"] = self._neq_As[i]
      data[f"C1_{i}"] = self._neq_C1s[i]
      data[f"T1_{i}"] = self._neq_T1s[i]
      data[f"C2_{i}"] = self._neq_C2s[i]
      data[f"T2_{i}"] = self._neq_T2s[i]
      data[f"C3_{i}"] = self._neq_C3s[i]
      data[f"T3_{i}"] = self._neq_T3s[i]
      data[f"C4_{i}"] = self._neq_C4s[i]
      data[f"T4_{i}"] = self._neq_T4s[i]
      data[f"colors_C1_r_{i}"] = self._colors_C1_r[i]
      data[f"colors_C1_d_{i}"] = self._colors_C1_d[i]
      data[f"colors_C2_d_{i}"] = self._colors_C2_d[i]
      data[f"colors_C2_l_{i}"] = self._colors_C2_l[i]
      data[f"colors_C3_u_{i}"] = self._colors_C3_u[i]
      data[f"colors_C3_l_{i}"] = self._colors_C3_l[i]
      data[f"colors_C4_u_{i}"] = self._colors_C4_u[i]
      data[f"colors_C4_r_{i}"] = self._colors_C4_r[i]
      for l in range(6):
        data[f"colors_A_{i}_{l}"] = self._colors_A[i][l]

    np.savez_compressed(saveFile, **data)


  def load_from_file(self, saveFile):
    """
    Load cell, tensors and colors from saveFile. Erase any pre-existing data.
    """
    with np.load(saveFile) as data:
      self._cell = data["cell"]
      self._indices = data["indices"]
      self._neq_coords = data["neq_coords"]
      self._Ly, self._Lx = self._cell.shape
      self._Nneq = len(self._neq_coords)

      self._neq_As = [None]*self._Nneq
      self._neq_C1s = [None]*self._Nneq
      self._neq_T1s = [None]*self._Nneq
      self._neq_C2s = [None]*self._Nneq
      self._neq_T2s = [None]*self._Nneq
      self._neq_C3s = [None]*self._Nneq
      self._neq_T3s = [None]*self._Nneq
      self._neq_C4s = [None]*self._Nneq
      self._neq_T4s = [None]*self._Nneq

      # colors are always defined and stored, even if they are default_color
      self._colors_A = [(default_color,)*6]*self._Nneq
      self._colors_C1_r = [default_color]*self._Nneq
      self._colors_C1_d = [default_color]*self._Nneq
      self._colors_C2_d = [default_color]*self._Nneq
      self._colors_C2_l = [default_color]*self._Nneq
      self._colors_C3_u = [default_color]*self._Nneq
      self._colors_C3_l = [default_color]*self._Nneq
      self._colors_C4_u = [default_color]*self._Nneq
      self._colors_C4_r = [default_color]*self._Nneq

      for i in range(self._Nneq):
        self._neq_As[i] = data[f"A_{i}"]
        self._neq_C1s[i] = data[f"C1_{i}"]
        self._neq_T1s[i] = data[f"T1_{i}"]
        self._neq_C2s[i] = data[f"C2_{i}"]
        self._neq_T2s[i] = data[f"T2_{i}"]
        self._neq_C3s[i] = data[f"C3_{i}"]
        self._neq_T3s[i] = data[f"T3_{i}"]
        self._neq_C4s[i] = data[f"C4_{i}"]
        self._neq_T4s[i] = data[f"T4_{i}"]
        self._colors_A[i] = tuple(data[f"colors_A_{i}_{l}"] for l in range(6))
        self._colors_C1_r[i] = data[f"colors_C1_r_{i}"]
        self._colors_C1_d[i] = data[f"colors_C1_d_{i}"]
        self._colors_C2_d[i] = data[f"colors_C2_d_{i}"]
        self._colors_C2_l[i] = data[f"colors_C2_l_{i}"]
        self._colors_C3_u[i] = data[f"colors_C3_u_{i}"]
        self._colors_C3_l[i] = data[f"colors_C3_l_{i}"]
        self._colors_C4_u[i] = data[f"colors_C4_u_{i}"]
        self._colors_C4_r[i] = data[f"colors_C4_r_{i}"]

    self._reset_projectors_temp()


  def set_tensors(self, tensors, colors=None):
    if self._Nneq != len(tensors):
     raise ValueError("Incompatible cell and tensors")
    for i,A in enumerate(tensors):  # or self._neq_coords?
      A = np.ascontiguousarray(A)
      if A.ndim == 5:  # if no ancila, add 1
        A = A.reshape(A.shape[0],1,A.shape[1],A.shape[2],A.shape[3],A.shape[4])
      oldA = self._neq_As[i]
      oldcol = self._colors_A[i]
      if colors is None:
        col = (default_color,)*6
      else:
        col = colors[i]
        if len(col) == 5:
          col = (col[0],np.zeros(1,dtype=np.int8),*col[1:])
        if tuple(len(c) for c in col) != A.shape:
          raise ValueError("Colors do not match tensors")
      if A.shape[0] != oldA.shape[0] or A.shape[1] != oldA.shape[1] or (col[0] != oldcol[0]).any() or (col[1] != oldcol[1]).any():
        # not a problem for the code, but physically meaningless
        raise ValueError('Physical and ancila dimensions and colors cannot change')

      # when the shape of a tensor (or its quantum numbers) changes, we still
      # would like to keep the environment. This is still possible: the legs of
      # A were obtained from some SVD then truncated, ie some singular value
      # was put to 0 (either in former or current A). Just add a row of zeros
      # corresponding to this 0 singular value in the tensor that misses it and
      # dimensions match. In case of U(1) symmetry, this is the same but sector
      # wise.

      x,y = self._neq_coords[i]
      # up axis
      if A.shape[2] != oldA.shape[2] or (oldcol[2]!=col[2]).any():
        j = self._indices[x%self._Lx, (y-1)%self._Ly]
        oldT1 = self._neq_T1s[j]
        newT1 = np.zeros((oldT1.shape[0],A.shape[2],A.shape[2],oldT1.shape[2]))
        if oldcol[2].size:   # colorwise copy
          old_rows, new_rows = _color_correspondence(oldcol[2], col[2])  # put copy outside of color loop
        else:   # colors are not provided
          old_rows = slice(0,oldA.shape[2])
          new_rows = slice(0,A.shape[2])
        newT1[:,new_rows,new_rows] = oldT1[:,old_rows,old_rows]
        self._neq_T1s[j] = newT1

      # right axis
      if A.shape[3] != oldA.shape[3] or (oldcol[3]!=col[3]).any():
        j = self._indices[(x+1)%self._Lx, y%self._Ly]
        oldT2 = self._neq_T2s[j]
        newT2 = np.zeros((oldT2.shape[0],oldT2.shape[1],A.shape[3],A.shape[3]))
        if oldcol[3].size:   # colorwise copy
          old_rows, new_rows = _color_correspondence(oldcol[3], col[3])
        else:   # colors are not provided
          old_rows = slice(0,oldA.shape[3])
          new_rows = slice(0,A.shape[3])
        newT2[:,:,new_rows,new_rows] = oldT2[:,:,old_rows,old_rows]
        self._neq_T2s[j] = newT2

      if A.shape[4] != oldA.shape[4] or (oldcol[4]!=col[4]).any():
        j = self._indices[x%self._Lx, (y+1)%self._Ly]
        oldT3 = self._neq_T3s[j]
        newT3 = np.zeros((A.shape[4],A.shape[4],oldT3.shape[2],oldT3.shape[3]))
        if oldcol[4].size:   # colorwise copy
          old_rows, new_rows = _color_correspondence(oldcol[4], col[4])
        else:   # colors are not provided
          old_rows = slice(0,oldA.shape[4])
          new_rows = slice(0,A.shape[4])
        newT3[new_rows,new_rows] = oldT3[old_rows,old_rows]
        self._neq_T3s[j] = newT3

      if A.shape[5] != oldA.shape[5] or (oldcol[5]!=col[5]).any():
        j = self._indices[(x-1)%self._Lx, y%self._Ly]
        oldT4 = self._neq_T4s[j]
        newT4 = np.zeros((oldT4.shape[0],A.shape[5],A.shape[5],oldT4.shape[3]))
        if oldcol[5].size:   # colorwise copy
          old_rows, new_rows = _color_correspondence(oldcol[5], col[5])
        else:   # colors are not provided
          old_rows = slice(0,oldA.shape[5])
          new_rows = slice(0,A.shape[5])
        newT4[:,new_rows,new_rows] = oldT4[:,old_rows,old_rows]
        self._neq_T4s[j] = newT4

      self._neq_As[i] = A
      self._colors_A[i] = col


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

  def get_tensor_type(self,x,y):
    return self._cell[x%self._Lx, y%self._Ly]

  def get_A(self,x,y):
    return self._neq_As[self._indices[x%self._Lx, y%self._Ly]]

  def get_colors_A(self,x,y):
    return self._colors_A[self._indices[x%self._Lx, y%self._Ly]]

  def get_tensor_type(self,x,y):
    return self._cell[x%self._Lx, y%self._Ly]

  def get_C1(self,x,y):
    return self._neq_C1s[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_C1_r(self,x,y):
    return self._colors_C1_r[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_C1_d(self,x,y):
    return self._colors_C1_d[self._indices[x%self._Lx, y%self._Ly]]

  def get_T1(self,x,y):
    return self._neq_T1s[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_T1_r(self,x,y):
    return -self._colors_C2_l[self._indices[(x+1)%self._Lx, y%self._Ly]]

  def get_color_T1_d(self,x,y):
    return -self._colors_A[self._indices[x%self._Lx, (y+1)%self._Ly]][2]

  def get_color_T1_l(self,x,y):
    return -self._colors_C1_r[self._indices[(x-1)%self._Lx, y%self._Ly]]

  def get_C2(self,x,y):
    return self._neq_C2s[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_C2_d(self,x,y):
    return self._colors_C2_d[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_C2_l(self,x,y):
    return self._colors_C2_l[self._indices[x%self._Lx, y%self._Ly]]

  def get_T2(self,x,y):
    return self._neq_T2s[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_T2_u(self,x,y):
    return -self._colors_C2_d[self._indices[x%self._Lx, (y-1)%self._Ly]]

  def get_color_T2_d(self,x,y):
    return -self._colors_C3_u[self._indices[x%self._Lx, (y+1)%self._Ly]]

  def get_color_T2_l(self,x,y):
    return -self._colors_A[self._indices[(x-1)%self._Lx, y%self._Ly]][3]

  def get_C3(self,x,y):
    return self._neq_C3s[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_C3_u(self,x,y):
    return self._colors_C3_u[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_C3_l(self,x,y):
    return self._colors_C3_l[self._indices[x%self._Lx, y%self._Ly]]

  def get_T3(self,x,y):
    return self._neq_T3s[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_T3_u(self,x,y):
    return -self._colors_A[self._indices[x%self._Lx, (y-1)%self._Ly]][4]

  def get_color_T3_r(self,x,y):
    return -self._colors_C3_l[self._indices[(x+1)%self._Lx, y%self._Ly]]

  def get_color_T3_l(self,x,y):
    return -self._colors_C4_r[self._indices[(x-1)%self._Lx, y%self._Ly]]

  def get_C4(self,x,y):
    return self._neq_C4s[self._indices[x%self._Lx,y%self._Ly]]

  def get_color_C4_u(self,x,y):
    return self._colors_C4_u[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_C4_r(self,x,y):
    return self._colors_C4_r[self._indices[x%self._Lx, y%self._Ly]]

  def get_T4(self,x,y):
    return self._neq_T4s[self._indices[x%self._Lx,y%self._Ly]]

  def get_color_T4_u(self,x,y):
    return -self._colors_C1_d[self._indices[x%self._Lx, (y-1)%self._Ly]]

  def get_color_T4_r(self,x,y):
    return -self._colors_A[self._indices[(x+1)%self._Lx, y%self._Ly]][5]

  def get_color_T4_d(self,x,y):
    return -self._colors_C4_u[self._indices[x%self._Lx, (y+1)%self._Ly]]

  def get_P(self,x,y):
    return self._neq_P[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_P(self,x,y):
    return self._colors_P[self._indices[x%self._Lx, y%self._Ly]]

  def get_Pt(self,x,y):
    return self._neq_Pt[self._indices[x%self._Lx, y%self._Ly]]

  def get_color_Pt(self,x,y):
    return self._colors_Pt[self._indices[x%self._Lx, y%self._Ly]]

  def _reset_projectors_temp(self):
    # free projectors memory, other arrays are stored anyway. Is it worth it?
    self._neq_P = [None]*self._Nneq
    self._neq_Pt = [None]*self._Nneq
    self._nCX = [None]*self._Nneq
    self._nT = [None]*self._Nneq
    self._nCY = [None]*self._Nneq
    self._colors_P = [None]*self._Nneq
    self._colors_CX = [None]*self._Nneq
    self._colors_CY = [None]*self._Nneq

  def store_projectors(self,x,y,P,Pt,color_P=default_color):
    j = self._indices[x%self._Lx, y%self._Ly]
    self._neq_P[j] = P
    self._neq_Pt[j] = Pt
    self._colors_P[j] = color_P

  def store_renormalized_tensors(self,x,y,nCX,nT,nCY,color_P=default_color,color_Pt=default_color):
    j = self._indices[x%self._Lx, y%self._Ly]
    self._nCX[j] = nCX
    self._nT[j] = nT
    self._nCY[j] = nCY
    self._colors_CX[j] = color_P
    self._colors_CY[j] = color_Pt

  def fix_renormalized_up(self):
    self._neq_C1s = self._nCX
    self._neq_T1s = self._nT
    self._neq_C2s = self._nCY
    self._colors_C1_r = self._colors_CX
    self._colors_C2_l = self._colors_CY
    self._reset_projectors_temp()

  def fix_renormalized_right(self):
    self._neq_C2s = self._nCX
    self._neq_T2s = self._nT
    self._neq_C3s = self._nCY
    self._colors_C2_d = self._colors_CX
    self._colors_C3_u = self._colors_CY
    self._reset_projectors_temp()

  def fix_renormalized_down(self):
    self._neq_C3s = self._nCX
    self._neq_T3s = self._nT
    self._neq_C4s = self._nCY
    self._colors_C3_l = self._colors_CX
    self._colors_C4_r = self._colors_CY
    self._reset_projectors_temp()

  def fix_renormalized_left(self):
    self._neq_C4s = self._nCX
    self._neq_T4s = self._nT
    self._neq_C1s = self._nCY
    self._colors_C4_u = self._colors_CX
    self._colors_C1_d = self._colors_CY
    self._reset_projectors_temp()
