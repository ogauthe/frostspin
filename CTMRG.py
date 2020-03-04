import numpy as np
import scipy.linalg as lg
from Env import Env
from ctm_contract import construct_U_half, construct_L_half, construct_D_half, construct_R_half

class CTMRG(object):
  #
  #    C1-T1-T1-C4
  #    |  |  |  |
  #    T2-a--a--T4
  #    |  |  |  |
  #    T2-a--a--T4
  #    |  |  |  |
  #    C2-T3-T3-C3

  def __init__(self,tensors,tiling,chi,verbosity=0):
    self.verbosity = verbosity
    self.chi = chi
    if self.verbosity > 0:
      print(f'initalize CTMRG with chi = {chi}, verbosity = {verbosity} and tiling = {tiling}')
    self._env = Env(tensors,tiling,chi)
    self._neq_coords = self._env.neq_coords
    self._Nneq = len(self._neq_coords)
    self._D = tensors[0].shape[1]
    self._D2 = self._D**2
    if self.verbosity > 0:
      print('CTMRG constructed')

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
    if self.verbosity > 0:
      print('start left move')
    # 1) compute isometries for every non-equivalent sites
    P_list,Pt_list = [None]*self._Nneq, [None]*self._Nneq
    for x,y in self._neq_coords:
      j = self._env.get_neq_index(x,y+1)
      #      UUUU     UUUU     0
      #      |  |  => |  |  => R
      #      0  1     1  0     1
      R = construct_U_half(self._env,x,y).T
      #      1  0     0  1     0
      #      |  |  => |  |  => Rt
      #      DDDD     DDDD     1
      Rt = construct_D_half(self._env,x,y).T
      P_list[j], Pt_list[j] = self.construct_projectors(R,Rt)

    # 2) renormalize every non-equivalent C1, T2 and C2
    # P != P_list[i] => need all projectors to be constructed at this time
    del R, Rt
    nC1s,nT2s,nC2s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x+1,y)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      Pt = Pt_list[i]
      P = P_list[self._env.get_neq_index(x,y-1)]
      nC1s[j] = self.renormalize_C1(x,y,Pt)
      nT2s[j] = self.renormalize_T2(x,y,P,Pt)
      nC2s[j] = self.renormalize_C2(x,y,P)

    # 3) store renormalized tensors in the environment
    # read C1[x,y] but write C1[x+1,y] => need a separate loop
    self._env.neq_C1s = nC1s
    self._env.neq_T2s = nT2s
    self._env.neq_C2s = nC2s

    if self.verbosity > 0:
      print('left move completed')


  def construct_projectors(self,R,Rt):
    U,s,V = lg.svd(R @ Rt)
    U_H = U[:,:self.chi].T.conj()
    V_H = V[:self.chi].T.conj()
    s12 = 1/np.sqrt(s[:self.chi])
    #  0        0
    #  ||       |
    #  Pt       P
    #  |        ||
    #  1        1
    Pt = Rt @ np.einsum('ij,j->ij', V_H, s12)
    P = np.einsum('ij,i->ij', U_H, s12) @ R
    return P,Pt

  def renormalize_C1(self,x,y,Pt):
    """
    Renormalize corner C1 using projector Pt
    CPU: 2*chi**3*D**2
    """
    # contracting T1 and Pt first set complexity to chi^3*(D^2+chi).
    # no leading order gain for chi<D^2, do not implement it.

    #                      0
    #  C1-1    0-T1-2      ||
    #  |         |         Pt
    #  0         1         |
    #                      1
    C1 = self._env.get_C1(x,y)
    T1 = self._env.get_T1(x+1,y)
    if self.verbosity > 0:
      print('Renormalize C1.\nC1.shape =', C1.shape, 'T1.shape =', T1.shape,
            'Pt.shape =',Pt.shape)

    #  C1-10-T1-2 ->1
    #  |     |
    #  0     1
    #    \ /
    #     0
    nC1 = np.tensordot(C1,T1,((1,),(0,))).reshape(len(Pt),T1.shape[2])
    #  C1--T1-1
    #  |   |
    #   \ /
    #    0
    #    0
    #    ||
    #    Pt
    #    |
    #    1 ->0
    nC1 = Pt.T @ nC1
    return nC1

  def renormalize_T2(self,x,y,P,Pt):
    """
    Renormalize edge T2 using projectord P and Pt
    CPU: chi**2*D**4*(2*chi+D**4)
    """
    #    0          0        0         0
    #    |          ||       |         |
    #    T2-2       Pt     1-a-3       P
    #    |          |        |         ||
    #    1          1        2         1
    T2 = self._env.get_T2(x,y)
    a = self._env.get_a(x+1,y)
    if self.verbosity > 0:
      print('Renormalize T2.\nT2.shape =', T2.shape, 'P.shape =', P.shape,
            'Pt.shape =',Pt.shape)

    #       0    # reshape P to 3-leg tensor
    #       |
    #       P3
    #     /   \
    #     1    2
    P3 = P.reshape(len(P),T2.shape[0],self._D2)
    #       0
    #       |
    #       P3
    #     /   \
    #     1    2 ->1
    #     0
    #     |
    #     T2-2 ->3
    #     |
    #     1 -> 2
    nT2 = np.tensordot(P3,T2,((1,),(0,)))

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
    #     T2----a-3 ->2
    #     |     |
    #     1     2
    #      \  /
    #        1
    nT2 = nT2.reshape(len(P),len(Pt),self._D2)

    #        0
    #        |
    #        P
    #      /   \
    #     |     |
    #     |     |
    #     |     |
    #     T2----a-2 -> 1 -> 2
    #      |    |
    #       \  /
    #        1
    #        0
    #        ||
    #        Pt
    #        |
    #        1 -> 2 -> 1
    nT2 = np.tensordot(nT2,Pt,((1),(0,))).swapaxes(1,2)
    return nT2

  def renormalize_C2(self,x,y,P):
    """
    Renormalize corner C2 using projector P
    CPU: 2*chi**3*D**2
    """
    #  0         0         0
    #  |         |         |
    #  C2-1    1-T3-2      P
    #                     ||
    #                      1
    C2 = self._env.get_C2(x,y)
    T3 = self._env.get_T3(x+1,y)
    if self.verbosity > 0:
      print('Renormalize C2.\nC2.shape =', C2.shape, 'T3.shape =', T3.shape,
            'P.shape =',P.shape)

    #     0     0 ->1
    #     |     |
    #     C2-11-T3-2
    nC2 = np.tensordot(C2,T3,((1,),(1,)))

    #        0
    #       /  \
    #     0     1
    #     |     |
    #     C2----T3-2 -> 1
    nC2 = nC2.reshape(P.shape[1],T3.shape[2])

    #        0
    #        |
    #        P
    #       ||
    #        1
    #        0
    #       /  \
    #     0     1
    #     |     |
    #     C2----T3-1
    nC2 = P @ nC2
    return nC2


