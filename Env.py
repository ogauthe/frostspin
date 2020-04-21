import numpy as np
import scipy.linalg as lg


def initialize_env(A,chi):
  #
  #   C1-0  2-T1-0  1-C2            0
  #   |       |        |             \ 1
  #   1       1        0              \|
  #                                  4-A-2
  #   0       0        0               |
  #   |       |        |               3
  #   T4-1  3-a--1  2-T2
  #   |       |        |
  #   2       2        1
  #
  #   0       0        0
  #   |       |        |
  #   C4-1  2-T3-1  1-C3
  #
  D = A.shape[1]   # do not consider the case Dx != Dy
  a = np.tensordot(A,A.conj(),(0,0)).transpose(0,4,1,5,2,6,3,7).reshape(D**2,D**2,D**2,D**2)
  C1 = np.einsum('iijkll->jk', a.reshape(D,D,D**2,D**2,D,D))
  T1 = np.einsum('iijkl->jkl', a.reshape(D,D,D**2,D**2,D**2))
  C2 = np.einsum('iijjkl->kl', a.reshape(D,D,D,D,D**2,D**2))
  T2 = np.einsum('ijjkl->ikl', a.reshape(D**2,D,D,D**2,D**2))
  C3 = np.einsum('ijjkkl->il', a.reshape(D**2,D,D,D,D,D**2))
  T3 = np.einsum('ijkkl->ijl', a.reshape(D**2,D**2,D,D,D**2))
  C4 = np.einsum('ijkkll->ij', a.reshape(D**2,D**2,D,D,D,D))
  T4 = np.einsum('ijkll->ijk', a.reshape(D**2,D**2,D**2,D,D))
  return a,C1,T1,T2,C2,T3,C3,T4,C4


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

  def __init__(self, tensors, tiling, chi):
    """
    tensors: list-like containing tensors.
    tiling: string. Tiling pattern.
    """
    self._tensors = list(tensors)
    tiling1 = tiling.strip()
    letters = list(tiling1.replace('\n','').replace(' ',''))
    tensors_str = sorted(set(letters))
    self._Lx = tiling1.index('\n')
    self._Ly = len(letters)//self._Lx
    indices = np.array([tensors_str.index(t) for t in letters])
    self._cell = np.array(list(map(chr,indices+65))).reshape(self._Lx,self._Ly)
    self._Nneq = len(tensors_str)
    if self._Nneq != len(tensors):
      raise ValueError("incompatible tiling and tensors")

    self._neq_coords = np.empty((self._Nneq,2),dtype=np.int8)
    for i in range(self._Nneq):
      ind = np.argmax(indices==i)
      self._neq_coords[i] = ind//self._Ly, ind%self._Lx

    self._indices = indices.reshape(self._Lx,self._Ly)
    self._neq_as = []
    self._neq_C1s = []
    self._neq_T1s = []
    self._neq_C2s = []
    self._neq_T2s = []
    self._neq_C3s = []
    self._neq_T3s = []
    self._neq_C4s = []
    self._neq_T4s = []

    self._neq_As = np.ascontiguousarray(tensors)
    for A in tensors:
      a,C1,T1,C2,T2,C3,T3,C4,T4 = initialize_env(A,chi)
      self._neq_as.append(a)
      self._neq_C1s.append(C1)
      self._neq_T1s.append(T1)
      self._neq_C2s.append(C2)
      self._neq_T2s.append(T2)
      self._neq_C3s.append(C3)
      self._neq_T3s.append(T3)
      self._neq_C4s.append(C4)
      self._neq_T4s.append(T4)

    self.reset_projectors()

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
  def indices(self):
    return self._indices

  @property
  def neq_coords(self):
    return self._neq_coords

  def get_neq_index(self,x,y):
    return self._indices[x%self._Lx, y%self._Ly]

  def get_A(self,x,y):
    return self._neq_As[self._indices[x%self._Lx, y%self._Ly]]

  def get_a(self,x,y):
    return self._neq_as[self._indices[x%self._Lx, y%self._Ly]]

  def get_tensor_type(self,x,y):
    return self._cell[x%self._Lx, y%self._Ly]

  def get_C1(self,x,y):
    return self._neq_C1s[self._indices[x%self._Lx,y%self._Ly]]

  def get_T1(self,x,y):
    return self._neq_T1s[self._indices[x%self._Lx,y%self._Ly]]

  def get_C2(self,x,y):
    return self._neq_C2s[self._indices[x%self._Lx, y%self._Ly]]

  def get_T2(self,x,y):
    return self._neq_T2s[self._indices[x%self._Lx,y%self._Ly]]

  def get_C3(self,x,y):
    return self._neq_C3s[self._indices[x%self._Lx,y%self._Ly]]

  def get_T3(self,x,y):
    return self._neq_T3s[self._indices[x%self._Lx,y%self._Ly]]

  def get_C4(self,x,y):
    return self._neq_C4s[self._indices[x%self._Lx,y%self._Ly]]

  def get_T4(self,x,y):
    return self._neq_T4s[self._indices[x%self._Lx,y%self._Ly]]

  def get_P(self,x,y):
    return self._neq_P[self._indices[x%self._Lx,y%self._Ly]]

  def get_Pt(self,x,y):
    return self._neq_Pt[self._indices[x%self._Lx,y%self._Ly]]

  def reset_projectors(self):
    self._neq_P = [None]*self._Nneq
    self._neq_Pt = [None]*self._Nneq

  def set_projectors(self,x,y,P,Pt):
    j = self._indices[x%self._Lx, y%self._Ly]
    self._neq_P[j] = P
    self._neq_Pt[j] = Pt

  @property
  def neq_C1s(self):
    return self._neq_C1s

  @neq_C1s.setter
  def neq_C1s(self, neq_C1s):
    assert(len(neq_C1s) == self._Nneq), 'neq_C1s length is not nneq'
    self._neq_C1s = neq_C1s

  @property
  def neq_T1s(self):
    return self._neq_T1s

  @neq_T1s.setter
  def neq_T1(self, neq_T1s):
    assert(len(neq_T1s) == self._Nneq), 'neq_T1s length is not nneq'
    self._neq_T1s = neq_T1s

  @property
  def neq_C2s(self):
    return self._neq_C2s

  @neq_C2s.setter
  def neq_C2s(self, neq_C2s):
    assert(len(neq_C2s) == self._Nneq), 'neq_C2s length is not nneq'
    self._neq_C2s = neq_C2s

  @property
  def neq_T2s(self):
    return self._neq_T2s

  @neq_T2s.setter
  def neq_T2s(self, neq_T2s):
    assert(len(neq_T2s) == self._Nneq), 'neq_T2s length is not nneq'
    self._neq_T2s = neq_T2s

  @property
  def neq_C3s(self):
    return self._neq_C3s

  @neq_C3s.setter
  def neq_C3s(self, neq_C3s):
    assert(len(neq_C3s) == self._Nneq), 'neq_C3s length is not nneq'
    self._neq_C3s = neq_C3s

  @property
  def neq_C4s(self):
    return self._neq_C4s

  @neq_C4s.setter
  def neq_C4s(self, neq_C4s):
    assert(len(neq_C4s) == self._Nneq), 'neq_C4s length is not nneq'
    self._neq_C4s = neq_C4s

  @property
  def neq_T4s(self):
    return self._neq_T4s

  @neq_T4s.setter
  def neq_T4s(self, neq_T4s):
    assert(len(neq_T4s) == self._Nneq), 'neq_T4s length is not nneq'
    self._neq_T4s = neq_T4s
