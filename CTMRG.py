import numpy as np
import scipy.linalg as lg

class CTMRG(object):
  #
  #    C1-T1-T1-C4
  #    |  |  |  |
  #    T2-a--a--T4
  #    |  |  |  |
  #    T2-a--a--T4
  #    |  |  |  |
  #    C2-T3-T3-C3

  def __init__(self,tensors,tiling,chi):
    self._env = Env(tensors,tiling,chi)
    self._neq_coords = self._env.neq_coords
    self._Nneq = len(self._neq_coords)
    self.chi = chi
    self._D = tensors[0].shape[1]
    self._D2 = self._D**2

  @property
  def Lx(self):
    return self._env.Lx

  @property
  def Ly(self):
    return self._env.Ly

  @property
  def env(self):
    return self._env

  def iterate(self):
    self.up_move()
    self.left_move()
    self.down_move()
    self.right_move()

  def left_move(self):
    # 1) compute isometries for every non-equivalent sites
    P_list,Pt_list = [None]*self._Nneq, [None]*self._Nneq
    for x,y in self._neq_coords:
      j = self._env.get_neq_index(x,y+1)
      R = self.construct_U_half(x,y)
      Rt = self.construct_D_half(x,y)
      P_list[j], Pt_list[j] = self.construct_projectors(R,Rt)

    # 2) renormalize every non-equivalent C1, T2 and C2
    nC1s,nT2s,nC2s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x+1,y)
      Pt = Pt_list[i]
      P = P_list[self._env.get_neq_index(x,y-1)]
      nC1s[j] = self.renormalize_C1(x,y,Pt)
      nT2s[j] = self.renormalize_T2(x,y,P,Pt)
      nC2s[j] = self.renormalize_C1(x,y,P)

    # 3) store renormalized tensors in the environment
    self._env.neq_C1s = nC1s
    self._env.neq_T2s = nT2s
    self._env.neq_C2s = nC2s

  def renormalize_C1(self,x,y,Pt):
    """
    Renormalize corner C1 using projector Pt
    CPU: 2*chi**3*D**2
    """
    # contracting T1 and Pt first set complexity to chi^3*(D^2+chi).
    # no leading order gain for chi<D^2, do not implement it.

    #  C1-1    0-T1-2    1    0
    #  |         |        \Pt/
    #  0         1         |
    #                      2
    C1 = self._env.get_C1(x,y)
    T1 = self._env.get_T1(x+1,y)
    PtT = Pt.swapaxes(0,1)

    #  C1-10-T1-2
    #  |     |
    #  0     1
    nC1 = np.tensordot(C1,T1,((1,),(0,)))
    #  C1---T1-2 ->1
    #  |     |
    #  0     1
    #   1    0
    #    \Pt/
    #     |
    #     2 ->0
    nC1 = np.tensordot(PtT,nC1,((1,0),(0,1)))
    return nC1

  def renormalize_T2(self,x,y,P,Pt):
    """
    Renormalize edge T2 using projectord P and Pt
    CPU: chi**2*D**4*(2*chi+D**4)
    """
    #    0        1   0      0           0
    #    |         \ /       |           |
    #    T2-2       Pt     1-a-3         P
    #    |          |        |          / \
    #    1          2        2         1   2
    T2 = self._env.get_T2(x,y)
    a = self._env.get_a(x+1,y)
    PT = P.transpose(2,0,1)
    PtT = Pt.swapaxes(0,1)

    #       0
    #       |
    #       P
    #     /   \
    #     1    2 ->1
    #     0
    #     |
    #     T2-2 ->3
    #     |
    #     1 -> 2
    nT2 = np.tensordot(PT,T2,((1,),(0,)))

    #        0
    #        |
    #        P
    #      /   \
    #     |     1
    #     |     0
    #     |     |
    #     T2-31-a-3
    #     |     |
    #     2->1  2
    nT2 = np.tensordot(nT2,a,((1,3),(0,1)))

    #        0
    #        |
    #        P
    #      /   \
    #     |     |
    #     |     |
    #     |     |
    #     T2----a-3 ->1 -> 2
    #     |     |
    #     1     2
    #      1    0
    #       \  /
    #        Pt
    #        |
    #        2 ->2 -> 1
    nT2 = np.tensordot(nT2,PtT,((2,1),(0,1))).swapaxes(1,2)
    return nT2

  def renormalize_C2(self,x,y,P):
    """
    Renormalize corner C2 using projector P
    CPU: 2*chi**3*D**2
    """
    #  0         0         0
    #  |         |         |
    #  C2-1    1-T3-2      P
    #                     / \
    #                    1   2
    C2 = self._env.get_C2(x,y)
    T3 = self._env.get_T3(x+1,y)
    PT = P.transpose(2,0,1)

    #     0     0 ->1
    #     |     |
    #     C2-11-T3-2
    nC2 = np.tensordot(C2,T3,((1,),(1,)))

    #        0
    #        |
    #        P
    #       / \
    #      1   2
    #     0     1
    #     |     |
    #     C2----T3-2 ->1
    nC2 = np.tensordot(PT,nC2,((1,2),(0,1)))
    return nC2




###############################################################################
#  construct 2x2 corners
###############################################################################
  def construct_UL_corner(self,x,y):
    #                     0
    #                     |
    #  0-T1-2    C1-1     T2-2
    #    |       |        |
    #    1       0        1
    a = self._env.get_a(x+1,y+1)
    T1 = self._env.get_T1(x+1,y)
    C1 = self._env.get_C1(x,y)
    T2 = self._env.get_T2(x+2,y)

    #  C1-10-T1-2->1
    #  |     |
    #  0->2  1->0
    cornerUL = np.tensordot(T1,C1,((0,),(1,)))

    #  C1---T1-1
    #  |    |
    #  2    0
    #  0
    #  |
    #  T2-2 -> 3
    #  |
    #  1 -> 2
    cornerUL = np.tensordot(cornerUL,T2,((2,),(0,)))

    #  C1----T1-1 -> 0
    #  |     |
    #  |     0
    #  |     0
    #  |     |
    #  T2-31-a-3
    #  |     |
    #  2->1  2
    cornerUL = np.tensordot(cornerUL,a,((0,3),(0,1)))

    lx = T1.shape[2]*a.shape[3]
    ly = T2.shape[1]*a.shape[2]
    #  C1-T1-0->2\
    #  |  |       1
    #  T2-a-3->3 /
    #  |  |
    #  1  2
    #  0  1
    #  \ /
    #   0
    cornerUL = cornerUL.transpose(1,2,0,3).reshape(ly,lx)
    return cornerUL


  def construct_DL_corner(self,x,y):
    #    0           0             0
    #    |           |             |
    #    T2-2      1-T3-2          C2-1
    #    |
    #    1
    a = self._env.get_a(x+1,y+2)
    T2 = self._env.get_T2(x,y+2)
    C2 = self._env.get_C2(x,y+3)
    T3 = self._env.get_T3(x+1,y+3)

    #      0
    #      |
    #      T2-2 -> 1
    #      |
    #      1
    #      0
    #      |
    #      C2-0 -> 2
    cornerDL = np.tensordot(T2,C2,((1,),(0,)))

    #      0
    #      |
    #      T2-1
    #      |
    #      |
    #      |     0 -> 2
    #      |     |
    #      C2-21-T3-2 -> 3
    cornerDL = np.tensordot(cornerDL,T3,((2,),(1,)))

    #      0     0 -> 2
    #      |     |
    #      T2-11-a--3
    #      |     |
    #      |     2
    #      |     2
    #      |     |
    #      C2----T3-3 -> 1
    cornerDL = np.tensordot(cornerDL,a,((1,2),(1,2)))

    lx = T3.shape[2]*a.shape[3]
    ly = T2.shape[0]*a.shape[0]
    #       0
    #       /\
    #      0  1
    #      0  2
    #      |  |
    #      T2-a--33\
    #      |  |     1
    #      C2-T3-12/
    cornerDL = cornerDL.swapaxes(1,2).reshape(ly,lx)
    return cornerDL


  def construct_DR_corner(self,x,y):
    #    0           0             0
    #    |           |             |
    #  1-T4        1-T3-2        1-C3
    #    |
    #    2
    a = self._env.get_a(x+2,y+2)
    T3 = self._env.get_T3(x+2,y+3)
    C3 = self._env.get_C3(x+3,y+3)
    T4 = self._env.get_T4(x+3,y+2)

    #       0     0->2
    #       |     |
    #     1-T3-21-C3
    cornerDR = np.tensordot(T3,C3,((2,),(1,)))

    #             0->2
    #             |
    #       3 <-1-T4
    #             |
    #             2
    #       0     2
    #       |     |
    #     1-T3----C3
    cornerDR = np.tensordot(cornerDR,T4,((2,),(2,)))

    #       0    2->3
    #       |    |
    #     1-a-33-T4
    #       |    |
    #       2    |
    #       0    |
    #       |    |
    #  2<-1-T3---C3
    cornerDR = np.tensordot(a,cornerDR,((2,3),(0,3)))

    ly = T4.shape[0]*a.shape[0]
    lx = T3.shape[1]*a.shape[1]
    #        0
    #        /\
    #       1  0
    #       0  3
    #       |  |
    #   /31-a--T4
    #  1    |  |
    #   \22-T3-C3
    cornerDR = cornerDR.transpose(3,0,2,1).reshape(ly,lx)
    return cornerDR


  def construct_UR_corner(self,x,y):
    #               0
    #               |
    #  0-T1-2     1-T4      0-C4
    #    |          |         |
    #    1          2         1
    a = self._env.get_a(x+2,y+1)
    T4 = self._env.get_T4(x+3,y+1)
    C4 = self._env.get_C4(x+3,y)
    T1 = self._env.get_T1(x+2,y)


    #  0-T1-20-C4
    #    |     |
    #    1     1->2
    cornerUR = np.tensordot(T1,C4,((2,),(0,)))

    #  0-T1----C4
    #    |     |
    #    1     2
    #          0
    #          |
    #     2<-1-T4
    #          |
    #          2-> 3
    cornerUR = np.tensordot(cornerUR,T4,((2,),(0,)))

    #    0-T1---C4
    #      |    |
    #      1    |
    #      0    |
    #      |    |
    # 2<-1-a-32-T4
    #      |    |
    #  3 <-2    3 -> 1
    cornerUR = np.tensordot(cornerUR,a,((1,2),(0,3)))

    lx = T1.shape[0]*a.shape[1]
    ly = T4.shape[2]*a.shape[2]
    #    /00-T1-C4
    #   0    |  |
    #    \12-a--T4
    #        |  |
    #        3  1
    #        3  2
    #         \/
    #         1
    cornerUR = cornerUR.swapaxes(1,2).reshape(lx,ly)
    return cornerUR


###############################################################################
# construct halves from corners
# again do not reshape anything
###############################################################################

  def construct_U_half(self,x,y):
    cornerUL = self.construct_UL_corner(x,y)
    cornerUR = self.construct_UR_corner(x,y)
    #  UL-10-UR
    #  |     |
    #  0     1
    return cornerUL @ cornerUR


  def construct_L_half(self,x,y):
    cornerUL = self.construct_UL_corner(x,y)
    cornerDL = self.construct_DL_corner(x,y)
    # UL-1
    # |
    # 0
    # 0
    # |
    # DL-1 ->0
    return cornerDL.T @ cornerUL


  def construct_D_half(self,x,y):
    cornerDL = self.construct_DL_corner(x,y)
    cornerDR = self.construct_DR_corner(x,y)
    #  1     0
    #  0     1
    #  |     |
    #  DL-10-DR
    return (cornerDL @ cornerDR).T


  def construct_R_half(self,x,y):
    cornerDR = self.construct_DR_corner(x,y)
    cornerUR = self.construct_UR_corner(x,y)
    #   0-DR
    #     |
    #     1
    #     0
    #     |
    #   1-UR
    return cornerDR @ cornerUR


  def construct_projectors(self,R,Rt):
    U,s,V = lg.svd(R.T @ Rt)
    U_H = U[:,:self.chi].T.conj()
    V_H = V[:self.chi].T.conj()
    s12 = 1/np.sqrt(s[:self.chi])
    #   ||    <- size chi*D**2
    #    R
    #    |
    #    V
    #    |
    #    s  <- size chi
    Pt = (Rt @ np.einsum('ij,j->ij', V_H, s12)).reshape(self.chi,self._D2,self.chi)
    P = (np.einsum('ij,i->ij', U_H, s12)@ R).reshape(self.chi,self._D2,self.chi)
    return P,Pt


def initialize_env(A,chi):
    D = A.shape[1]   # do not consider the case Dx != Dy
    a = np.tensordot(A,A.conj(),(0,0)).transpose(0,4,1,5,2,6,3,7).copy()
    T1 = np.einsum('iijkl->jkl', a.reshape(D,D,D**2,D**2,D**2))
    C1 = np.einsum('iijjkl->kl', a.reshape(D,D,D,D,D**2,D**2))
    T2 = np.einsum('ijjkl->ikl', a.reshape(D**2,D,D,D**2,D**2))
    C2 = np.einsum('ijjkkl->il', a.reshape(D**2,D,D,D,D,D**2))
    T3 = np.einsum('ijkkl->ijl', a.reshape(D**2,D**2,D,D,D**2))
    C3 = np.einsum('ijkkll->ij', a.reshape(D**2,D**2,D,D,D,D))
    T4 = np.einsum('ijkll->ijk', a.reshape(D**2,D**2,D**2,D,D))
    C4 = np.einsum('iijkll->jk', a.reshape(D,D,D**2,D**2,D,D))
    a = a.reshape(D**2,D**2,D**2,D**2)
    return a,T1,C1,T2,C2,T3,C3,T4,C4



class Env(object):
  """
  Container for CTMRG environment tensors.
  leg conventions:

     C1-T1-T1-C4
     |  |  |  |
     T2-a--a--T4
     |  |  |  |
     T2-a--a--T4
     |  |  |  |
     C2-T3-T3-C3
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
    self._neq_T1s = []
    self._neq_C1s = []
    self._neq_T2s = []
    self._neq_C2s = []
    self._neq_T3s = []
    self._neq_C3s = []
    self._neq_T4s = []
    self._neq_C4s = []

    for A in tensors:
      a,T1,C1,T2,C2,T3,C3,T4,C4 = initialize_env(A,chi)
      self._neq_as.append(a)
      self._neq_T1s.append(T1)
      self._neq_C1s.append(C1)
      self._neq_T2s.append(T2)
      self._neq_C2s.append(C2)
      self._neq_T3s.append(T3)
      self._neq_C3s.append(C3)
      self._neq_T4s.append(T4)
      self._neq_C4s.append(C4)

    # 1st renormaliztion without absorbtion
    for i,(x,y) in enumerate(self._neq_coords):
      iT1 = self._indices[x,(y-1)%self._Ly]
      iC1 = self._indices[(x-1)%self._Lx,(y-1)%self._Ly]
      iT2 = self._indices[(x-1)%self._Lx,y]
      iC2 = self._indices[(x-1)%self._Lx,(y+1)%self._Ly]
      iT3 = self._indices[x,(y+1)%self._Ly]
      iC3 = self._indices[(x+1)%self._Lx,(y+1)%self._Ly]
      iT4 = self._indices[(x+1)%self._Lx,y]
      iC4 = self._indices[(x+1)%self._Lx,(y-1)%self._Ly]
      #   s-V-T1
      #   |
      #   U
      #   |
      #   T2
      U,s,V = lg.svd(self._neq_C1s[iC1])
      self._neq_T1s[iT1] = np.tensordot(V[:chi],self._neq_T1s[iT1],((1,),(0,)))
      self._neq_C1s[iC1] = np.diag(s[:chi])
      self._neq_T2s[iT2] = np.tensordot(U[:,:chi],self._neq_T2s[iT2],((0,),(0,)))
      #   T2
      #   |
      #   U
      #   |
      #   s-V-T3
      U,s,V = lg.svd(self._neq_C2s[iC2])
      self._neq_T2s[iT2] = np.tensordot(self._neq_T2s[iT2],U[:,:chi],((1,),(0,))).swapaxes(1,2)
      self._neq_C2s[iC2] = np.diag(s[:chi])
      self._neq_T3s[iT3] = np.tensordot(V[:chi],self._neq_T3s[iT3],((1,),(1,))).swapaxes(0,1)
      #       T4
      #       |
      #       U
      #       |
      #  T3-V-s
      U,s,V = lg.svd(self._neq_C3s[iC3])
      self._neq_T3s[iT3] = np.tensordot(self._neq_T3s[iT3],V[:chi],((2,),(1,)))
      self._neq_C3s[iC3] = np.diag(s[:chi])
      self._neq_T4s[iT4] = np.tensordot(self._neq_T4s[iT4],U[:,:chi],((2,),(0,)))
      #    T1-U-s
      #         |
      #         V
      #         |
      #         T4
      U,s,V = lg.svd(self._neq_C4s[iC4])
      self._neq_T4s[iT4] = np.tensordot(V[:chi],self._neq_T4s[iT4],((1,),(0,)))
      self._neq_C4s[iC4] = np.diag(s[:chi])
      self._neq_T1s[iT1] = np.tensordot(self._neq_T1s[iT1],U[:,:chi],((2,),(0,)))



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

  def get_a(self,x,y):
    return self._neq_as[self._indices[x%self._Lx, y%self._Ly]]

  def get_tensor_type(self,x,y):
    return self._cell[x%self._Lx, y%self._Ly]

  def get_T1(self,x,y):
    return self._neq_T1s[self._indices[x%self._Lx,y%self._Ly]]

  def get_C1(self,x,y):
    return self._neq_C1s[self._indices[x%self._Lx, y%self._Ly]]

  def get_T2(self,x,y):
    return self._neq_T2s[self._indices[x%self._Lx,y%self._Ly]]

  def get_C2(self,x,y):
    return self._neq_C2s[self._indices[x%self._Lx,y%self._Ly]]

  def get_T3(self,x,y):
    return self._neq_T3s[self._indices[x%self._Lx,y%self._Ly]]

  def get_C3(self,x,y):
    return self._neq_C3s[self._indices[x%self._Lx,y%self._Ly]]

  def get_T4(self,x,y):
    return self._neq_T4s[self._indices[x%self._Lx,y%self._Ly]]

  def get_C4(self,x,y):
    return self._neq_C4s[self._indices[x%self._Lx,y%self._Ly]]

  @property
  def neq_T1s(self):
    return self._neq_T1s

  @neq_T1s.setter
  def neq_T1(self, neq_T1s):
    assert(len(neq_T1s) == self._Nneq), 'neq_T1s length is not nneq'
    self._neq_T1s = neq_T1s

  @property
  def neq_C1s(self):
    return self._neq_C1s

  @neq_C1s.setter
  def neq_C1s(self, neq_C1s):
    assert(len(neq_C1s) == self._Nneq), 'neq_C1s length is not nneq'
    self._neq_C1s = neq_C1s

  @property
  def neq_T2s(self):
    return self._neq_T2s

  @neq_T2s.setter
  def neq_T2s(self, neq_T2s):
    assert(len(neq_T2s) == self._Nneq), 'neq_T2s length is not nneq'
    self._neq_T2s = neq_T2s

  @property
  def neq_C2s(self):
    return self._neq_C2s

  @neq_C2s.setter
  def neq_C2s(self, neq_C2s):
    assert(len(neq_C2s) == self._Nneq), 'neq_C2s length is not nneq'
    self._neq_C2s = neq_C2s

  @property
  def neq_T3s(self):
    return self._neq_T3s

  @neq_T3s.setter
  def neq_T3s(self, neq_T3s):
    assert(len(neq_T3s) == self._Nneq), 'neq_T3s length is not nneq'
    self._neq_T3s = neq_T3s

  @property
  def neq_C3s(self):
    return self._neq_C3s

  @neq_C3s.setter
  def neq_C3s(self, neq_C3s):
    assert(len(neq_C3s) == self._Nneq), 'neq_C3s length is not nneq'
    self._neq_C3s = neq_C3s

  @property
  def neq_T4s(self):
    return self._neq_T4s

  @neq_T4s.setter
  def neq_T4s(self, neq_T4s):
    assert(len(neq_T4s) == self._Nneq), 'neq_T4s length is not nneq'
    self._neq_T4s = neq_T4s

  @property
  def neq_C4s(self):
    return self._neq_C4s

  @neq_C4s.setter
  def neq_C4s(self, neq_C4s):
    assert(len(neq_C4s) == self._Nneq), 'neq_C4s length is not nneq'
    self._neq_C4s = neq_C4s
