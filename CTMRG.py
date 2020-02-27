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

  def __init__(self,Lx,Ly,A_i,chi):
    assert(len(A_i) == Lx), "A_i must be a list of list len (Lx,Ly)"
    assert(sum([len(a) == Ly for a in A_i)==Lx), "A_i must be a list of list len (Lx,Ly)"
    self._Lx = Lx
    self._Ly = Ly
    self._env = Env(Lx,Ly,A_i)
    self._chi = chi
    self._l = chi*D**2

  @property
  def Lx(self):
    return self._Lx

  @property
  def Ly(self):
    return self._Ly

  @property
  def chi(self):
    return self._chi

  @property
  def env(self):
    return self._env


  def do_step(self):
    for x in range(self._Lx):
      self.left_move(x)
      self.right_move(x)
    for y in range(self._Ly):
      self.up_move(y)
      self.down_move(y)


  def left_move(self,x,y):
    l = self._chi*self._D2
    # CT-TC
    # Ta-aT
    # || ||
    # 01 23
    # 01 32
    # |/ \|
    # 0   1
    halfU = self.construct_U_half(x)
    R = halfU.swapaxes(2,3).reshape(l,l)

    #    1   0
    #    |\ /|
    #    23 10
    #    32 10
    #    || ||
    #    Ta-aT
    #    CT-TC
    halfD = self.construct_L_half(x)
    Rt = halfD.swapaxes(2,3).reshape(l,l)

    P,Pt = self.construct_projectors(R,Rt)
    self._env.renormalize_C1(x,P)
    self._env.renormalize_T2(x,P,Pt)
    self._env.renormalize_C2(x,Pt)


  def right_move(self,x,y):
    l = self._chi*self._D2
    # CT-TC
    # Ta-aT
    # || ||
    # 01 23
    # 01 32
    # |/ \|
    # 0   1
    halfU = self.construct_U_half(x)
    Rt = halfU.swapaxes(2,3).reshape(l,l)

    #    1   0
    #    |\ /|
    #    23 10
    #    32 10
    #    || ||
    #    Ta-aT
    #    CT-TC
    halfD = self.construct_L_half(x)
    R = halfD.swapaxes(2,3).reshape(l,l)

    P,Pt = self.construct_projectors(R,Rt)
    self._env.renormalize_C4(x,P)
    self._env.renormalize_T4(x,P,Pt)
    self._env.renormalize_C3(x,Pt)


  def up_move(self,x,y):
    l = self._chi*self._D2
    # CT-00-0
    # Ta-11/
    # Ta-23\
    # CT-32-1
    halfL = self.construct_L_half(x,y)
    R = halfL.swapaxes(2,3).reshape(l,l)

    #  1-23-TC
    #   \32-aT
    #   /11-aT
    #  0-00-TC
    halfR = self.construct_R_half(x,y)
    Rt = halfR.swapaxes(2,3).reshape(l,l)

    P,Pt = self.construct_projectors(R,Rt)
    self._env.renormalize_C1(x,y,P)
    self._env.renormalize_T1(x,y,P,Pt)
    self._env.renormalize_C4(x,y,Pt)


  def down_move(self,x,y):
    l = self._chi*self._D2
    # CT-00-0
    # Ta-11/
    # Ta-23\
    # CT-32-1
    halfL = self.construct_L_half(x,y)
    Rt = halfL.swapaxes(2,3).reshape(l,l)

    #  1-23-TC
    #   \32-aT
    #   /11-aT
    #  0-00-TC
    halfR = self.construct_R_half(x,y)
    R = halfR.swapaxes(2,3).reshape(l,l)

    P,Pt = self.construct_projectors(R,Rt)
    self._env.renormalize_C2(x,y,P)
    self._env.renormalize_T3(x,y,P,Pt)
    self._env.renormalize_C3(x,y,Pt)



###############################################################################
#  construct 2x2 corners
#  transpose but do not reshape to keep view and avoid useless copy
#  another transpose will be made at half constructions
###############################################################################
  def construct_UL_corner(self,x,y):
    #               0
    #               |
    #  0-T1-2       T2-2      C1-1
    #    |          |         |
    #    1          1         0
    a = self._env.get_a(x+1,y+1)
    T1 = self._env.get_T1(x+1,y)
    T2 = self._env.get_T2(x+2,y)
    C1 = self._env.get_C1(x,y)

    #  C1-10-T1-2
    #  |     |
    #  0     1
    cornerUL = np.tensordot(C1,T1,(1,0))

    #  C1-10-T1-2 -> 1
    #  |     |
    #  0     1 -> 0
    #  0
    #  |
    #  T2-2 -> 3
    #  |
    #  1 -> 2
    cornerUL = np.tensordot(cornerUL,T2,(0,0))

    #  C1-10-T1-1 -> 0
    #  |     |
    #  0     0
    #  0     0
    #  |     |
    #  T2-31-a-3
    #  |     |
    #  2->1  2
    cornerUL = np.tensordot(cornerUL,a,((0,3),(0,1)))

    #  C1-10-T1-0 -> 3
    #  |     |
    #  0     0
    #  0     0
    #  |     |
    #  T2-31-a-3 -> 2
    #  |     |
    #  1->0  2 -> 1
    cornerUL = cornerUL.transpose(3,0,1,2)
    return cornerUL


  def construct_DL_corner(self,x,y):
    #    0           0             0
    #    |           |             |
    #    T2-2      1-T3-2          C2-1
    #    |
    #    1
    a = self._env.get_a(x+1,y+2)
    T2 = self._env.get_T2(x,y+2)
    T3 = self._env.get_T3(x+1,y+3)
    C2 = self._env.get_C2(x,y+3)

    #      0
    #      |
    #      T2-2 -> 1
    #      |
    #      1
    #      0
    #      |
    #      C2-0 -> 2
    cornerDL = np.tensordot(T2,C2,(1,0))

    #      0
    #      |
    #      T2-1
    #      |
    #      1
    #      0     0 -> 2
    #      |     |
    #      C2-21-T3-2 -> 3
    cornerDL = np.tensordot(cornerDL,T3,(2,1))

    #      0     0 -> 2
    #      |     |
    #      T2-11-a-3
    #      |     |
    #      1     2
    #      0     2
    #      |     |
    #      C2-21-T3-3 -> 1
    cornerDL = np.tensordot(cornerDL,a,((1,2),(1,2)))

    #      0->1  2 -> 0
    #      |     |
    #      T2-11-a-3
    #      |     |
    #      1     2
    #      0     2
    #      |     |
    #      C2-21-T3-1 -> 2
    cornerDL = cornerDL.transpose(2,0,1,3)
    return cornerDL


  def construct_DR_corner(self,x,y):
    #    0           0             0
    #    |           |             |
    #  1-T4        1-T3-2        1-C3
    #    |
    #    2
    a = self._env.get_a(x+2,y+2)
    T4 = self._env.get_T4(x+3,y+2)
    T3 = self._env.get_T3(x+2,y+3)
    C3 = self._env.get_C3(x+3,y+3)

    #       0     0->2
    #       |     |
    #     1-T3-21-C3
    cornerDR = np.tensordot(T3,C3,(2,1))

    #             0->2
    #             |
    #       3 <-1-T4
    #             |
    #             2
    #       0     2
    #       |     |
    #     1-T3-21-C3
    cornerDR = np.tensordot(cornerDR,T4,(2,2))

    #       0     2->3
    #       |     |
    #     1-a-3 3-T4
    #       |     |
    #       2     2
    #       0     2
    #       |     |
    #  2<-1-T3-21-C3
    cornerDR = np.tensordot(a,cornerDR,((2,3),(0,3)))

    #    1<-0     3->0
    #       |     |
    #  2<-1-a-3 3-T4
    #       |     |
    #       2     2
    #       0     2
    #       |     |
    #  3<-2-T3-21-C3
    cornerDR = cornerDR.transpose(3,0,1,2)
    return cornerDR


  def construct_UR_corner(self,x,y):
    #               0
    #               |
    #  0-T1-2     1-T4      0-C4
    #    |          |         |
    #    1          2         1
    a = self._env.get_a(x+2,y+1)
    T1 = self._env.get_T1(x+2,y)
    T4 = self._env.get_T4(x+3,y+1)
    C4 = self._env.get_C4(x+3,y)


    #  0-T1-20-C4
    #    |     |
    #    1     1->2
    cornerUR = np.tensordot(T1,C4,(2,0))

    #  0-T1-20-C4
    #    |     |
    #    1     2
    #          0
    #          |
    #     2<-1-T4
    #          |
    #          2-> 3
    cornerUR = np.tensordot(cornerUR,T4,(2,0))

    #    0-T1-20-C4
    #      |     |
    #      1     2
    #      0     0
    #      |     |
    # 2<-1-a-3 2-T4
    #      |     |
    #  3 <-2     3 -> 1
    cornerUR = np.tensordot(cornerUR,a,((1,2),(0,3)))

    #    0-T1-20-C4
    #      |     |
    #      1     2
    #      0     0
    #      |     |
    # 1<-2-a-3 2-T4
    #      |     |
    #  2 <-3     1 -> 3
    cornerUR = cornerUR.transpose(0,2,3,1)
    return cornerUR


###############################################################################
# construct halves from corners
# again do not reshape anything
###############################################################################

  def construct_U_half(self,x,y):
    cornerUL = self.construct_UL_corner(x,y)
    cornerUR = self.construct_UR_corner(x,y)

    # CT-3 0-TC
    # Ta-2 1-aT
    # ||     ||
    # 01     23
    halfU = np.tensorsot(cornerUL,cornerUR,((2,3),(1,0)))
    return halfU


  def construct_L_half(self,x,y):
    cornerUL = self.construct_UL_corner(x,y)
    cornerDL = self.construct_DL_corner(x,y)

    # CT-3
    # Ta-2
    # ||
    # 01
    # 10
    # ||
    # Ta-3 -> 1
    # CT-2 -> 0
    halfL = np.tensorsot(cornerDL,cornerUL,((1,0),(0,1)))
    return halfL


  def construct_D_half(self,x,y):
    cornerDL = self.construct_DL_corner(x,y)
    cornerDR = self.construct_DR_corner(x,y)

    #
    # 3<-10->2  10
    #    ||     ||
    #    Ta-3 2-aT
    #    CT-2 3-TC
    halfD = np.tensorsdot(cornerDR,cornerDL,((2,3),(3,2)))
    return halfD


  def construct_R_half(self,x,y):
    cornerDR = self.construct_DR_corner(x,y)
    cornerUR = self.construct_UR_corner(x,y)

    # 0-TC
    # 1-aT
    #   ||
    #   23
    #   10
    #   ||
    # 2-aT
    # 3-TC
    halfR = np.tensorsot(cornerUR,cornerDR,((2,3),(1,0)))
    return halfR


    def construct_projectors(self,R,Rt):
    M = R.T @ Rt
    U,s,V = lg.svd(M)
    s12 = 1/np.sqrt(s[:chi])
    Pt = np.einsum('ij,i->ij', V[:self._chi], s12) @ Rt
    P = R @ np.einsum('ij,j->ij', U[:,:self._chi], s12)
    return P,Pt



class Env(object):
  def __init__(self):
    self._nneq = nneq   # number of non-equivalent sites
    indices = np.empty((self._Lx, self._Ly), dtype=np.int8)
    self._indices = indices
    self._a = [None]*n_sites
    self._C1 = [None]*n_sites
    self._C2 = [None]*n_sites
    self._C3 = [None]*n_sites
    self._C4 = [None]*n_sites
    self._T1 = [None]*n_sites
    self._T2 = [None]*n_sites
    self._T3 = [None]*n_sites
    self._T4 = [None]*n_sites

  def get_a(self,x,y):
    return self._a[self._indices[x%self._Lx, y%self_Ly]]

  def getC1(self,x,y):
    return self._C1[self._indices[x%self._Lx, y%self_Ly]]

  def getC2(self,x,y):
    return self._C2[self._indices[x%self._Lx,y%self._Ly]]

  def getC3(self,x,y):
    return self._C3[self._indices[x%self._Lx,y%self._Ly]]

  def getC4(self,x,y):
    return self._C4[self._indices[x%self._Lx,y%self._Ly]]

  def getT1(self,x,y):
    return self._T1[self._indices[x%self._Lx,y%self._Ly]]

  def getT2(self,x,y):
    return self._T2[self._indices[x%self._Lx,y%self._Ly]]

  def getT3(self,x,y):
    return self._T3[self._indices[x%self._Lx,y%self._Ly]]

  def getT4(self,x,y):
    return self._T4[self._indices[x%self._Lx,y%self._Ly]]
