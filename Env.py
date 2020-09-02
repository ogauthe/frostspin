import numpy as np
from toolsU1 import default_color, combine_colors

def _cell_from_tiling(tiling):
  tiling1 = tiling.strip()
  raw_letters = list(tiling1.replace('\n','').replace(' ',''))
  try:
    n_rows = tiling1.index('\n')
    n_cols = len(raw_letters)//n_rows
  except ValueError:
    n_cols = len(tiling1)  # one line
    n_rows = 1

  letters_unique = sorted(set(raw_letters))
  indices = np.array([letters_unique.index(t) for t in raw_letters])
  cell = np.array(list(map(chr,indices+65))).reshape(n_rows,n_cols)
  return cell


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


class Env(object):
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

  def __init__(self, tensors, cell=None, tiling=None, colors=None):
    # could use dictionaty to deal properly with weird cells like AB//DC
    # but case is too unnatural to care about.
    """
    Store tensors and return them according to a given tiling.

    Parameters
    ----------
    tensors: list of Nneq numpy arrays
      Tensors given from left to right and up to down.
    cell: array of characters
      Elementary cell, each letter representing non-equivalent tensors. When
      following standard order, letters have to appear for the first by
      lexicographic order, else correspondance with tensors fails.
    tiling: string, optional.
      Tiling pattern. If cell is not provided, parse tiling to construct it.
    colors: list of Nneq colors
      U(1) quantum numbers corresponding to the tensors. Note that dimensions
      are check for compatibility with tensors, but color compatibility between
      legs to contract is not checked.
    """
    if cell is None:
      if tiling is None:
        raise ValueError("Either cell or tiling must be provided")
      cell = _cell_from_tiling(tiling)
    else:
      cell = np.asarray(cell)

    self._cell = cell
    # [row,col] indices are transposed from (x,y) coordinates
    self._Ly, self._Lx = cell.shape

    letters = sorted(set(cell.flat))
    self._Nneq = len(letters)
    if self._Nneq != len(tensors):
      raise ValueError("Inconpatible cell and tensors")

    self._neq_coords = np.empty((self._Nneq,2),dtype=np.int8)
    for i,l in enumerate(letters):
      self._neq_coords[i] = np.divmod((cell.flat == l).nonzero()[0][0],self._Lx)

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
      if len(colors) != len(tensors):
        raise ValueError("Colors numbers do not match tensors")
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
      self._colors_A = tuple(self._colors_A)
    else:
      self._colors_A = ((default_color,)*6,)*self._Nneq
      self._colors_C1_r = [default_color]*self._Nneq
      self._colors_C1_d = [default_color]*self._Nneq
      self._colors_C2_d = [default_color]*self._Nneq
      self._colors_C2_l = [default_color]*self._Nneq
      self._colors_C3_u = [default_color]*self._Nneq
      self._colors_C3_l = [default_color]*self._Nneq
      self._colors_C4_u = [default_color]*self._Nneq
      self._colors_C4_r = [default_color]*self._Nneq

    self._reset_projectors_temp()

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
    return self._neq_As[x%self._Lx + y%self._Ly*self._Lx]

  def get_colors_A(self,x,y):
    return self._colors_A[x%self._Lx + y%self._Ly*self._Lx]

  def get_C1(self,x,y):
    return self._neq_C1s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C1_r(self,x,y):
    return self._colors_C1_r[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C1_d(self,x,y):
    return self._colors_C1_d[x%self._Lx + y%self._Ly*self._Lx]

  def get_T1(self,x,y):
    return self._neq_T1s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_T1_r(self,x,y):
    return -self._colors_C2_l[(x+1)%self._Lx + y%self._Ly*self._Lx]

  def get_color_T1_d(self,x,y):
    return -self._colors_A[x%self._Lx + (y-1)%self._Ly*self._Lx][2]

  def get_color_T1_l(self,x,y):
    return -self._colors_C1_r[(x-1)%self._Lx + y%self._Ly*self._Lx]

  def get_C2(self,x,y):
    return self._neq_C2s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C2_d(self,x,y):
    return self._colors_C2_d[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C2_l(self,x,y):
    return self._colors_C2_l[x%self._Lx + y%self._Ly*self._Lx]

  def get_T2(self,x,y):
    return self._neq_T2s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_T2_u(self,x,y):
    return -self._colors_C2_d[x%self._Lx + (y+1)%self._Ly*self._Lx]

  def get_color_T2_d(self,x,y):
    return -self._colors_C3_u[x%self._Lx + (y-1)%self._Ly*self._Lx]

  def get_color_T2_l(self,x,y):
    return -self._colors_A[(x-1)%self._Lx + y%self._Ly*self._Lx][3]

  def get_C3(self,x,y):
    return self._neq_C3s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C3_u(self,x,y):
    return self._colors_C3_u[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C3_l(self,x,y):
    return self._colors_C3_l[x%self._Lx + y%self._Ly*self._Lx]

  def get_T3(self,x,y):
    return self._neq_T3s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_T3_u(self,x,y):
    return -self._colors_A[x%self._Lx + (y+1)%self._Ly*self._Lx][4]

  def get_color_T3_r(self,x,y):
    return -self._colors_C3_l[(x+1)%self._Lx + y%self._Ly*self._Lx]

  def get_color_T3_l(self,x,y):
    return -self._colors_C4_r[(x-1)%self._Lx + y%self._Ly*self._Lx]

  def get_C4(self,x,y):
    return self._neq_C4s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C4_u(self,x,y):
    return self._colors_C4_u[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_C4_r(self,x,y):
    return self._colors_C4_r[x%self._Lx + y%self._Ly*self._Lx]

  def get_T4(self,x,y):
    return self._neq_T4s[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_T4_u(self,x,y):
    return -self._colors_C1_d[x%self._Lx + (y+1)%self._Ly*self._Lx]

  def get_color_T4_r(self,x,y):
    return -self._colors_A[(x+1)%self._Lx + y%self._Ly*self._Lx][5]

  def get_color_T4_d(self,x,y):
    return -self._colors_C4_u[x%self._Lx + (y-1)%self._Ly*self._Lx]

  def get_P(self,x,y):
    return self._neq_P[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_P(self,x,y):
    return self._colors_P[x%self._Lx + y%self._Ly*self._Lx]

  def get_Pt(self,x,y):
    return self._neq_Pt[x%self._Lx + y%self._Ly*self._Lx]

  def get_color_Pt(self,x,y):
    return self._colors_Pt[x%self._Lx + y%self._Ly*self._Lx]

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
    j = x%self._Lx + y%self._Ly*self._Lx
    self._neq_P[j] = P
    self._neq_Pt[j] = Pt
    self._colors_P[j] = color_P

  def store_renormalized_tensors(self,x,y,nCX,nT,nCY,color_P=default_color,color_Pt=default_color):
    j = x%self._Lx + y%self._Ly*self._Lx
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
