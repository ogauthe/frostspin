import numpy as np
import scipy.linalg as lg
from Env import Env
from ctm_contract import construct_U_half, construct_L_half, construct_D_half, construct_R_half

class CTMRG(object):
  #
  #    C2-T1-T1-C1
  #    |  |  |  |
  #    T2-a--a--T4
  #    |  |  |  |
  #    T2-a--a--T4
  #    |  |  |  |
  #    C3-T3-T3-C4

  def __init__(self,tensors,tiling,chi,verbosity=0):
    self.verbosity = verbosity
    if self.verbosity > 0:
      print(f'initalize CTMRG with chi = {chi}, verbosity = {verbosity} and tiling = {tiling}')
    self.chi = chi
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

  def up_move(self):
    if self.verbosity > 0:
      print('\nstart up move')
    # 1) compute isometries for every non-equivalent sites
    # convention : for every move, leg 0 of R and Rt are to be contracted
    for x,y in self._neq_coords:
      #      0-R
      #        R
      #        R
      #      1-R
      R = construct_R_half(self._env,x,y)
      #        L-1         L-0
      #        L           L          0
      #        L   =>      L    =>    Rt
      #        L-0         L-1        1
      Rt = construct_L_half(self._env,x,y).T
      self._env.set_projectors(x+1,y,self.construct_projectors(R,Rt))
      del R, Rt

    # 2) renormalize every non-equivalent C1, T1 and C2
    # P != P_list[i] => need all projectors to be constructed at this time
    nC1s,nT1s,nC2s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x,y+1)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC1s[j] = self.renormalize_C1_up(x,y)
      nT1s[j] = self.renormalize_T1(x,y)
      nC2s[j] = self.renormalize_C2_up(x,y)

    # 3) store renormalized tensors in the environment
    # renormalization reads C2[x,y] but write C2[x,y+1]
    # => need to compute every renormalized tensors before storing any of them
    self._env.neq_C1s = nC1s
    self._env.neq_T1s = nT1s
    self._env.neq_C2s = nC2s

    if self.verbosity > 0:
      print('up move completed')

  def left_move(self):
    if self.verbosity > 0:
      print('\nstart left move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #      UUUU    0
      #      |  | => R
      #      0  1    1
      R = construct_U_half(self._env,x,y)
      #      1  0     0  1     0
      #      |  |  => |  |  => Rt
      #      DDDD     DDDD     1
      Rt = construct_D_half(self._env,x,y).T
      self._env.set_projectors(x,y+1,self.construct_projectors(R,Rt))
      del R, Rt

    # 2) renormalize tensors by absorbing column
    nC2s,nT2s,nC3s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x+1,y)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC2s[j] = self.renormalize_C2_left(x,y)
      nT2s[j] = self.renormalize_T2(x,y)
      nC3s[j] = self.renormalize_C3_left(x,y)

    # 3) store renormalized tensors in the environment
    self._env.neq_C2s = nC2s
    self._env.neq_T2s = nT2s
    self._env.neq_C3s = nC3s

    if self.verbosity > 0:
      print('left move completed')

  def down_move(self):
    if self.verbosity > 0:
      print('\nstart down move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #        L-1
      #        L
      #        L
      #        L-0
      R = construct_L_half(self._env,x,y)
      #      0-R         1-R
      #        R           R
      #        R   =>      R
      #      1-R         0-R
      Rt = construct_R_half(self._env,x,y).T
      self._env.set_projectors(x+1,y+3,self.construct_projectors(R,Rt))
      del R, Rt

    # 2) renormalize every non-equivalent C1, T1 and C2
    nC3s,nT3s,nC4s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x,y-1)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC3s[j] = self.renormalize_C3_down(x,y)
      nT3s[j] = self.renormalize_T3(x,y)
      nC4s[j] = self.renormalize_C4_down(x,y)

    # 3) store renormalized tensors in the environment
    self._env.neq_C3s = nC3s
    self._env.neq_T3s = nT3s
    self._env.neq_C4s = nC4s

    if self.verbosity > 0:
      print('down move completed')

  def right_move(self):
    if self.verbosity > 0:
      print('\nstart right move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #      1  0
      #      |  |
      #      DDDD
      R = construct_D_half(self._env,x,y)
      #      UUUU     UUUU     1
      #      |  |  => |  |  => Rt
      #      0  1     1  0     0
      Rt = construct_U_half(self._env,x,y).T
      self._env.set_projectors(x+3,y+1,self.construct_projectors(R,Rt))
      del R, Rt

    # 2) renormalize every non-equivalent C4, T4 and C1
    nC4s,nT4s,nC1s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x-1,y)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC4s[j] = self.renormalize_C4_right(x,y)
      nT4s[j] = self.renormalize_T4(x,y)
      nC1s[j] = self.renormalize_C1_right(x,y)

    # 3) store renormalized tensors in the environment
    self._env.reset_projectors()
    self._env.neq_C4s = nC4s
    self._env.neq_T4s = nT4s
    self._env.neq_C1s = nC1s

    if self.verbosity > 0:
      print('right move completed')

  def construct_projectors(self,R,Rt):
    if self.verbosity > 1:
      print('construct projectors: R.shape =',R.shape, 'Rt.shape =', Rt.shape)
    # convention : for every move, leg 0 of R and Rt are to be contracted
    U,s,V = lg.svd(R.T @ Rt)
    s12 = 1/np.sqrt(s[:self.chi])
    # convention: double-leg has index 0, single leg has index 1
    #  0        1
    #  ||       |
    #  Pt       P
    #  |        ||
    #  1        0
    Pt = Rt @ np.einsum('ij,i->ji', V[:self.chi].conj(), s12)
    P = R @ np.einsum('ij,j->ij', U[:,:self.chi].conj(), s12)
    return P,Pt

  def renormalize_C1_up(self,x,y):
    """
    Renormalize corner C1 from an up move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #                0
    #                |
    #       0-C1   1-T4    1-Pt=0
    #         |      |
    #         1      2
    C1 = self._env.get_C1(x,y)
    T4 = self._env.get_T4(x,y+1)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize C1: C1.shape =', C1.shape, 'T4.shape =', T4.shape,
            'Pt.shape =',Pt.shape)
    #      0-c1
    #     /   |
    #    /    1
    #    0    0
    #     \   |
    #      \1-t4
    #         |
    #         2 ->1
    nC1 = np.tensordot(C1,T4,(1,0)).reshape(len(Pt),T4.shape[2])
    #             /-C1
    #  0<- 1-Pt=00  |
    #             \-T4
    #               |
    #               2 -> 1
    nC1 = Pt.T @ nC1
    return nC1/lg.norm(nC1)

  def renormalize_T1(self,x,y):
    """
    Renormalize edge T1 using projectors P and Pt
    CPU: chi**2*D**4*(2*chi+D**4)
    """
    # contracting T1 and Pt first set complexity to chi^3*(D^2+chi).
    # no leading order gain for chi<D^2, do not implement it.
    #                       0
    #                       |
    #  1-Pt=0    0-T1-2   1-a-3     0=Pt-1
    #              |        |
    #              1        2
    T1 = self._env.get_T1(x,y)
    a = self._env.get_a(x,y+1)
    P = self._env.get_P(x+1,y)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize T1: T1.shape =', T1.shape, 'P.shape =', P.shape,
            'Pt.shape =',Pt.shape)
    #     0\       # reshape P to 3-leg tensor
    #       P-2
    #     1/
    P3 = P.reshape(T1.shape[2],self._D2,P.shape[1])
    #  0-T1-20\
    #    |     P-2 -> 3
    #    1    /
    #        1 ->2
    nT1 = np.tensordot(T1,P3,((2,),(0,)))
    #      0-T1--\
    #        |    P-3 ->1
    #        1   /
    #        0  /
    #        | 2
    #  2<-1-a-3
    #       |
    #       2 ->3
    nT1 = np.tensordot(nT1,a,((1,2),(0,3)))
    #     /--0-T1\
    #    0     |  P-1 ->3 ->2
    #    \1<-2-a-/
    #          |
    #          3 ->2 ->1
    nT1 = nT1.transpose(0,2,3,1).reshape(len(Pt),self._D2,Pt.shape[1])
    #             /T1\
    #  0<- 1-Pt=00 |  P-2
    #             \a-/
    #              |
    #              1
    nT1 = np.tensordot(Pt,nT1,((0,),(0,)))
    return nT1/lg.norm(nT1)

  def renormalize_C2_up(self,x,y):
    """
    Renormalize corner C2 from an up move using projector P
    CPU: 2*chi**3*D**2
    """
    #            0
    #            |
    #  C2-1      T2-2    0=P-1
    #  |         |
    #  0         1
    C2 = self._env.get_C2(x,y)
    T2 = self._env.get_T2(x,y+1)
    P = self._env.get_Pt(x+1,y)
    if self.verbosity > 0:
      print('Renormalize C2: C2.shape =', C2.shape, 'T2.shape =', T2.shape,
            'P.shape =',P.shape)

    #  C2-1 ->0
    #  |
    #  0
    #  0
    #  |
    #  T2-2
    #  |
    #  1
    nC2 = np.tensordot(C2,T2,((0,),(0,)))
    #  C2-0----\
    #  |        0
    #  T2-2 ->1/
    #  |
    #  1 -> 2 -> 1
    nC2 = nC2.swapaxes(1,2).reshape(len(P),T2.shape[1])
    #  C2-\
    #  |   00=P-1
    #  T2-/
    #  |
    #  1 ->0
    nC2 = nC2.T @ P
    return nC2/lg.norm(nC2)

  def renormalize_C2_left(self,x,y):
    """
    Renormalize corner C2 from a left move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #                      0
    #  C2-1    0-T1-2      ||
    #  |         |         Pt
    #  0         1         |
    #                      1
    C2 = self._env.get_C2(x,y)
    T1 = self._env.get_T1(x+1,y)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize C2: C2.shape =', C2.shape, 'T1.shape =', T1.shape,
            'Pt.shape =',Pt.shape)

    #  C2-10-T1-2 ->1
    #  |     |
    #  0     1
    #    \ /
    #     0
    nC2 = np.tensordot(C2,T1,((1,),(0,))).reshape(len(Pt),T1.shape[2])
    #  C2--T1-1
    #  |   |
    #   \ /
    #    0
    #    0
    #    ||
    #    Pt
    #    |
    #    1 ->0
    nC2 = Pt.T @ nC2
    return nC2/lg.norm(nC2)

  def renormalize_T2(self,x,y):
    """
    Renormalize edge T2 using projectors P and Pt
    CPU: chi**2*D**4*(2*chi+D**4)
    """
    #    0          0        0         1
    #    |          ||       |         |
    #    T2-2       Pt     1-a-3       P
    #    |          |        |         ||
    #    1          1        2         0
    T2 = self._env.get_T2(x,y)
    a = self._env.get_a(x+1,y)
    P = self._env.get_P(x,y-1)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize T2: T2.shape =', T2.shape, 'P.shape =', P.shape,
            'Pt.shape =',Pt.shape)

    #       2    # reshape P to 3-leg tensor
    #       |
    #       P3
    #     /   \
    #     0    1
    P3 = P.reshape(len(T2),self._D2,P.shape[1])
    #       2 -> 1
    #       |
    #       P3
    #     /   \
    #     0    1 ->0
    #     0
    #     |
    #     T2-2 ->3
    #     |
    #     1 -> 2
    nT2 = np.tensordot(P3,T2,((0,),(0,)))

    #        1  -> 0
    #        |
    #        P
    #      /   \
    #     |     0
    #     |     0
    #     |     |
    #     T2-31-a-3
    #     |     |
    #     2->1  2
    nT2 = np.tensordot(nT2,a,((0,3),(0,1)))

    #        0
    #        |
    #        P
    #      /   \
    #     T2----a-3 ->2
    #     |     |
    #     1     2
    #      \  /
    #        1
    nT2 = nT2.reshape(P.shape[1],len(Pt),self._D2)

    #        0
    #        |
    #        P
    #      /   \
    #     T2----a-2 -> 1 -> 2
    #       \  /
    #        1
    #        0
    #        ||
    #        Pt
    #        |
    #        1 -> 2 -> 1
    nT2 = np.tensordot(nT2,Pt,((1),(0,))).swapaxes(1,2)
    return nT2/lg.norm(nT2)

  def renormalize_C3_left(self,x,y):
    """
    Renormalize corner C3 from a left move using projector P
    CPU: 2*chi**3*D**2
    """
    #  0         0         1
    #  |         |         |
    #  C3-1    1-T3-2      P
    #                     ||
    #                      0
    C3 = self._env.get_C3(x,y)
    T3 = self._env.get_T3(x+1,y)
    P = self._env.get_P(x,y-1)
    if self.verbosity > 0:
      print('Renormalize C3: C3.shape =', C3.shape, 'T3.shape =', T3.shape,
            'P.shape =',P.shape)

    #     0     0 ->1
    #     |     |
    #     C3-11-T3-2
    nC3 = np.tensordot(C3,T3,((1,),(1,)))

    #        0
    #       /  \
    #     0     1
    #     |     |
    #     C3----T3-2 -> 1
    nC3 = nC3.reshape(len(P),T3.shape[2])

    #        1 ->0
    #        |
    #        P
    #       ||
    #        0
    #        0
    #       /  \
    #     C3----T3-1
    nC3 = P.T @ nC3
    return nC3/lg.norm(nC3)

  def renormalize_C3_down(self,x,y):
    """
    Renormalize corner C3 from a down move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #            0
    #            |
    #  0         T2-2
    #  |         |         0=Pt-1
    #  C3-1      1
    C3 = self._env.get_C3(x,y)
    T2 = self._env.get_T2(x,y-1)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize C3: C3.shape =', C3.shape, 'T2.shape =', T2.shape,
            'Pt.shape =',Pt.shape)

    #   0
    #   |
    #   T2-2 ->1
    #   |
    #   1
    #   0
    #   |
    #   C3-1 ->2
    nC3 = np.tensordot(T2,C3,((1,),(0,)))
    #   0
    #   |
    #   T2-1 ->2\
    #   |        1
    #   C3-2 ->1/
    nC3 = nC3.swapaxes(1,2).reshape(len(T2),len(Pt))


    #   0
    #   |
    #   T2\
    #   |  10=Pt-1
    #   C3/
    nC3 = nC3 @ Pt
    return nC3/lg.norm(nC3)

  def renormalize_T3(self,x,y):
    """
    Renormalize edge T3 using projectors P and Pt
    CPU: chi**2*D**4*(2*chi+D**4)
    """
    #                    0
    #                    |
    #     0   1-P=0    1-a-3    0=Pt-1
    #     |              |
    #   1-T3-2           2
    T3 = self._env.get_T3(x,y)
    a = self._env.get_a(x,y-1)
    P = self._env.get_P(x-1,y)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize T3: T3.shape =', T3.shape, 'P.shape =', P.shape,
            'Pt.shape =',Pt.shape)

    #             /1
    #       2-P3=0    # reshape P to 3-leg tensor
    #             \0
    P3 = P.reshape(T3.shape[1],a.shape[1],P.shape[1])
    #     0<- 1
    #        /   0 ->2
    # 1<- 2-P    |
    #        \01-T3-2 ->3
    nT3 = np.tensordot(P3,T3,((0,),(1,)))
    #             0
    #             |
    #           1-a-3 ->1
    #          0  |
    #         /   2
    #        /    2
    # 2<- 1-P     |
    #        \----T3-3
    nT3 = np.tensordot(a,nT3,((1,2),(0,2)))
    #             0
    #             |
    #           /-a--1 ->3\
    #    1<- 2-P  |        2
    #           \-T3-3 ->2/
    nT3 = nT3.transpose(0,2,3,1).reshape(len(a),P.shape[1],len(Pt))

    #      0
    #      |
    #     /a-\
    #  1-P |  20=Pt-1 ->2
    #     \T3/
    nT3 = np.tensordot(nT3,Pt,((2,),(0,)))
    return nT3/lg.norm(nT3)

  def renormalize_C4_down(self,x,y):
    """
    Renormalize corner C4 from down move using projector P
    CPU: 2*chi**3*D**2
    """
    #    0         0
    #    |         |
    #  1-C4      1-T4      1-P=0
    #              |
    #              2
    C4 = self._env.get_C4(x,y)
    T4 = self._env.get_T4(x,y-1)
    P = self._env.get_P(x-1,y)
    if self.verbosity > 0:
      print('Renormalize C4: C4.shape =', C4.shape, 'T4.shape =', T4.shape,
            'P.shape =',P.shape)
    #        0
    #        |
    #      1-T4
    #        |
    #        2
    #        0
    #        |
    #  2<- 1-C4
    nC4 = np.tensordot(T4,C4,((2,),(0,)))

    #         0
    #         |
    # 1-  2<-1-T4
    #  \      |
    #    1<-2-C4
    nC4 = nC4.swapaxes(1,2).reshape(len(T4),len(P))

    #          0
    #          |
    #        /-T4
    #  1-P=01  |
    #        \-C4
    nC4 = nC4 @ P
    return nC4/lg.norm(nC4)

  def renormalize_C4_right(self,x,y):
    """
    Renormalize corner C4 from right move using projector Pt
    CPU: 2*chi**3*D**2
    """
    #    0         0            1
    #    |         |            |
    #  1-C4      1-T3-2         Pt
    #                           ||
    #                           0
    C4 = self._env.get_C4(x,y)
    T3 = self._env.get_T3(x-1,y)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize C4: C4.shape =', C4.shape, 'T3.shape =', T3.shape,
            'Pt.shape =',Pt.shape)
    #   1<- 0     0
    #       |     |
    # 2<- 1-T3-21-C4
    nC4 = np.tensordot(C4,T3,((1,),(2)))
    #          0
    #        /  \
    #        1  0
    #        |  |
    #  1<- 2-T3-C4
    nC4 = nC4.reshape(len(Pt),T3.shape[1])
    #           1 ->0
    #           |
    #           Pt
    #           ||
    #           0
    #           0
    #         /  \
    #       1-T3-C4
    nC4 = Pt.T @ nC4
    return nC4/lg.norm(nC4)

  def renormalize_T4(self,x,y):
    """
    Renormalize edge T4 using projectors P and Pt
    CPU: chi**2*D**4*(2*chi+D**4)
    """
    #  1       0         0       0
    #  |       |         |       ||
    #  Pt    1-T4      1-a-3     P
    #  ||      |         |       |
    #  0       2         2       1
    T4 = self._env.get_T4(x,y)
    a = self._env.get_a(x-1,y)
    P = self._env.get_P(x,y+1)
    Pt = self._env.get_Pt(x,y)
    if self.verbosity > 0:
      print('Renormalize T4: T4.shape =', T4.shape, 'P.shape =', P.shape,
            'Pt.shape =',Pt.shape)
    #         2
    #         |
    #         Pt    # reshape Pt to 3-leg tensor
    #       /   \
    #       1    0
    Pt3 = Pt.reshape(len(T4),len(a),Pt.shape[1])
    #         2 ->1
    #         |
    #         Pt
    #       /   \
    #   0<- 1    0
    #            0
    #            |
    #      2<- 1-T4
    #            |
    #        3<- 2
    nT4 = np.tensordot(Pt3,T4,((0,),(0,)))
    #          1 ->0
    #          |
    #          Pt
    #        /   \
    #        0    |
    #        0    |
    #        |    |
    #  2<- 1-a-32-T4
    #        |    |
    #    3<- 2    3 ->1
    nT4 = np.tensordot(nT4,a,((0,2),(0,3)))
    #         0
    #         |
    #         Pt
    #        / \
    # 1<- 2-a---T4
    #       |   |
    #       3   1
    #       3   2
    #        \  /
    #         2
    nT4 = nT4.swapaxes(1,2).reshape(Pt.shape[1],a.shape[1],len(P))
    #         0
    #         |
    #         Pt
    #        / \
    #     1-a---T4
    #        \  /
    #         2
    #         0
    #         ||
    #         P
    #         |
    #         1 -> 2
    nT4 = np.tensordot(nT4,P,((2,),(0,)))
    return nT4/lg.norm(nT4)

  def renormalize_C1_right(self,x,y):
    """
    Renormalize corner C1 from right move using projector P
    CPU: 2*chi**3*D**2
    """
    #                      0
    #                      ||
    #  0-C1     0-T1-2     P
    #    |        |        |
    #    1        1        1
    C1 = self._env.get_C1(x,y)
    T1 = self._env.get_T1(x-1,y)
    P = self._env.get_P(x,y+1)
    if self.verbosity > 0:
      print('Renormalize C1: C1.shape =', C1.shape, 'T1.shape =', T1.shape,
            'P.shape =',P.shape)
    #  0-T1-20-C1
    #    |     |
    #    1     1 ->2
    nC1 = np.tensordot(T1,C1,((2,),(0)))
    #  120-T1-C1
    #      |  |
    #      1  2
    #      1  0
    #       \/
    #       0
    nC1 = nC1.swapaxes(0,2).reshape(len(P),len(T1))
    #  0<- 1-T1-C1
    #         \/
    #          0
    #          0
    #          ||
    #          P
    #          |
    #          1
    nC1 = nC1.T @ P
    return nC1/lg.norm(nC1)

