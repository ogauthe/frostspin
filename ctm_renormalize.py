import numpy as np
import scipy.linalg as lg


def construct_projectors(R,Rt,chi,verbosity=0):
  assert(R.shape[0] == Rt.shape[0])
  if verbosity > 1:
    print('construct projectors: R.shape =',R.shape, 'Rt.shape =', Rt.shape)
  # convention : for every move, leg 0 of R and Rt are to be contracted
  U,s,V = lg.svd(R.T @ Rt)
  s12 = 1/np.sqrt(s[:chi])
  # convention: projectors have shape (last_chi*D**2,chi)
  #  00'      1
  #  ||       |
  #  Pt       P
  #  |        ||
  #  1        00'
  Pt = (Rt @ np.einsum('ij,i->ji', V[:chi].conj(), s12))
  P = (R @ np.einsum('ij,j->ij', U[:,:chi].conj(), s12))
  return P,Pt


def renormalize_C1_up(env,x,y,verbosity=0):
  """
  Renormalize corner C1 from an up move using projector P
  CPU: 2*chi**3*D**2
  """
  #            0
  #            |
  #  C1-0      T4-1    0=P-1
  #  |         |
  #  1         2
  T4 = env.get_T4(x,y+1)
  C1 = env.get_C1(x,y)
  P = env.get_P(x+1,y)
  if verbosity > 1:
    print('Renormalize C1: T4.shape =', T4.shape, 'C1.shape =', C1.shape,
          'P.shape =',P.shape)
  #  C1-0
  #  |
  #  1
  #  0
  #  |
  #  T4-1
  #  |
  #  2
  nC1 = np.tensordot(C1,T4,((1,),(0,)))
  #  C1-0\
  #  |    0
  #  T4-1/
  #  |
  #  2 -> 1
  nC1 = nC1.reshape(len(P),T4.shape[2])
  #  C1-\
  #  |   00=P-1 ->0
  #  T4-/
  #  |
  #  1
  nC1 = P.T @ nC1
  return nC1/lg.norm(nC1)


def renormalize_T1(env,x,y,verbosity=0):
  """
  Renormalize edge T1 using projectors P and Pt
  CPU: chi**2*D**4*(2*chi+D**4)
  """
  # contracting T1 and Pt first set complexity to chi^3*(D^2+chi).
  # no leading order gain for chi<D^2, do not implement it.
  #
  #      0  2-T1-0  0
  #     /      |      \
  #    /       1       \
  # 1-Pt       0        P-1
  #    \       |       /
  #     \0'  3-a-1  0'/
  #            |
  #            2
  T1 = env.get_T1(x,y)
  P = env.get_P(x+1,y)
  a = env.get_a(x,y+1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize T1: T1.shape =', T1.shape, 'P.shape =', P.shape,
          'Pt.shape =',Pt.shape)
  # automatically generated code
  last_chi = T1.shape[2]
  D2 = a.shape[2]
  chi = Pt.shape[1]
  P_ = P.reshape(last_chi,D2,chi).transpose(1, 2, 0).reshape(D2*chi, last_chi)
  nT1 = np.dot(P_,T1.reshape(last_chi, D2*last_chi)).reshape(D2, chi, D2, last_chi)
  del P_
  nT1 = nT1.transpose(1, 3, 2, 0).reshape(chi*last_chi, D2**2)
  nT1 = np.dot(nT1,a.reshape(D2**2, D2**2)).reshape(chi, last_chi, D2, D2)
  nT1 = nT1.transpose(0, 2, 1, 3).reshape(D2*chi, Pt.shape[0])
  nT1 = np.dot(nT1,Pt).reshape(chi, D2, chi)
  return nT1/lg.norm(nT1)


def renormalize_C2_up(env,x,y,verbosity=0):
  """
  Renormalize corner C2 from an up move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #      0  1-C2
  #     /      |
  #    /       0
  # 1-Pt       0
  #    \       |
  #     \0' 2-T2
  #            |
  #            1
  C2 = env.get_C2(x,y)
  T2 = env.get_T2(x,y+1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize C2: C2.shape =', C2.shape, 'T2.shape =', T2.shape,
          'Pt.shape =',Pt.shape)
  #       0-C2 -> transpose
  #          |
  #          1
  #          0
  #          |
  #       2-T2
  #          |
  #          1
  nC2 = np.tensordot(C2.T,T2,((1,),(0,)))
  #       10-C2
  #      /    |
  #     1     |
  #      \22-T2
  #           |
  #           1 ->0
  nC2 = np.tensordot(C2.T,T2,((1,),(0,))).swapaxes(0,1).reshape(T2.shape[1],Pt.shape[0])
  #             /-C2
  #      1-Pt=01   |
  #             \-T2
  #                |
  #                0
  nC2 = nC2 @ Pt
  return nC2/lg.norm(nC2)


def renormalize_T2(env,x,y,verbosity=0):
  """
  Renormalize edge T2 using projectors P and Pt
  CPU: chi**2*D**4*(2*chi+D**4)
  """
  T2 = env.get_T2(x,y)
  a = env.get_a(x-1,y)
  P = env.get_P(x,y+1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize T2: T2.shape =', T2.shape, 'P.shape =', P.shape,
          'Pt.shape =',Pt.shape)
  #       1
  #       |
  #       Pt
  #     /   \
  #    0'    0
  #    0     0
  #    |     |
  #  3-a-12-T2
  #    |     |
  #    2     1
  #    0'    0
  #     \   /
  #       P
  #       |
  #       1
  last_chi = T2.shape[0]
  D2 = a.shape[0]
  chi = P.shape[1]
  Pt_ = Pt.reshape(last_chi,D2,chi).transpose(1, 2, 0).reshape(D2*chi, last_chi)
  nT2 = np.dot(Pt_,T2.reshape(last_chi, D2*last_chi)).reshape(D2, chi*last_chi, D2)
  del Pt_
  nT2 = nT2.transpose(1, 0, 2).reshape(chi*last_chi, D2**2)
  nT2 = np.dot(nT2,a.reshape(D2**2, D2**2)).reshape(chi, P.shape[0], D2)
  nT2 = nT2.swapaxes(1,2).reshape(D2*chi, P.shape[0])
  nT2 = np.dot(nT2,P).reshape(chi, D2, chi)
  nT2 = nT2.swapaxes(1,2)/lg.norm(nT2)
  return nT2


def renormalize_C2_right(env,x,y,verbosity=0):
  """
  Renormalize corner C2 from right move using projector P
  CPU: 2*chi**3*D**2
  """
  #                      0
  #                      ||
  #  1-C2     2-T1-0     P
  #     |       |        |
  #     0       1        1
  C2 = env.get_C2(x,y)
  T1 = env.get_T1(x-1,y)
  P = env.get_P(x,y+1)
  if verbosity > 1:
    print('Renormalize C2: C2.shape =', C2.shape, 'T1.shape =', T1.shape,
          'P.shape =',P.shape)
  #   1<- 2-T1-01-C2
  #         |      |
  #         1      0
  #          \   /
  #            0
  nC2 = np.tensordot(C2,T1,((1,),(0))).reshape(len(P),T1.shape[2])
  #      1-T1-C2
  #         \/
  #          0
  #          0
  #          ||
  #          P
  #          |
  #          1
  nC2 = P.T @ nC2
  return nC2/lg.norm(nC2)


def renormalize_C3_right(env,x,y,verbosity=0):
  """
  Renormalize corner C3 from right move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #     0        0            1
  #     |        |            |
  #  1-C3      2-T3-1         Pt
  #                           ||
  #                           0
  C3 = env.get_C3(x,y)
  T3 = env.get_T3(x-1,y)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize C3: C3.shape =', C3.shape, 'T3.shape =', T3.shape,
          'Pt.shape =',Pt.shape)
  #   1<- 0      0
  #       |      |
  #     2-T3-11-C3
  nC3 = np.tensordot(C3,T3,((1,),(1)))
  #          0
  #        /  \
  #        1  0
  #        |  |
  #  1<- 2-T3-C3
  nC3 = nC3.reshape(len(Pt),T3.shape[1])
  #           1 ->0
  #           |
  #           Pt
  #           ||
  #           0
  #           0
  #         /  \
  #       1-T3-C3
  nC3 = Pt.T @ nC3
  return nC3/lg.norm(nC3)


def renormalize_C3_down(env,x,y,verbosity=0):
  """
  Renormalize corner C3 from down move using projector P
  CPU: 2*chi**3*D**2
  """
  #     0         0
  #     |         |
  #  1-C3      2-T2      1-P=0
  #               |
  #               1
  C3 = env.get_C3(x,y)
  T2 = env.get_T2(x,y-1)
  P = env.get_P(x-1,y)
  if verbosity > 1:
    print('Renormalize C3: C3.shape =', C3.shape, 'T2.shape =', T2.shape,
          'P.shape =',P.shape)
  #         0
  #         |
  #  1<- 2-T2
  #         |
  #         1
  #         0
  #         |
  #  2<- 1-C3
  nC3 = np.tensordot(T2,C3,((1,),(0,)))
  #           0
  #           |
  #  1- 2<-1-T2
  #   \       |
  #     1<-2-C3
  nC3 = nC3.swapaxes(1,2).reshape(len(T2),len(P))
  #          0
  #          |
  #        /-T2
  #  1-P=01  |
  #        \-C3
  nC3 = nC3 @ P
  return nC3/lg.norm(nC3)


def renormalize_T3(env,x,y,verbosity=0):
  """
  Renormalize edge T3 using projectors P and Pt
  CPU: chi**2*D**4*(2*chi+D**4)
  """


  #             0
  #             |
  #         0'3-a-10'
  #        /    |    \
  #     1-P     2     Pt-1
  #        \    0     /
  #         \   |    /
  #          02-T3-10
  T3 = env.get_T3(x,y)
  a = env.get_a(x,y-1)
  P = env.get_P(x-1,y)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize T3: T3.shape =', T3.shape, 'P.shape =', P.shape,
          'Pt.shape =',Pt.shape)

  last_chi = T3.shape[1]
  D2 = a.shape[0]
  chi = Pt.shape[1]
  nT3 = np.dot(T3.reshape(D2*last_chi, last_chi),P.reshape(last_chi, D2*chi)).reshape(D2, last_chi, D2, chi)
  nT3 = nT3.transpose(0,2,1,3).reshape(D2**2,chi*last_chi)
  nT3 = np.dot(a.reshape(D2**2,D2**2),nT3).reshape(D2,D2,last_chi,chi)
  nT3 = nT3.transpose(0,3,2,1).reshape(D2*chi,last_chi*D2)
  nT3 = np.dot(nT3,Pt).reshape(D2,chi,chi).swapaxes(1,2)
  return nT3/lg.norm(nT3)


def renormalize_C4_down(env,x,y,verbosity=0):
  """
  Renormalize corner C4 from a down move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #            0
  #            |
  #  0         T4-1
  #  |         |         0=Pt-1
  #  C4-1      2
  C4 = env.get_C4(x,y)
  T4 = env.get_T4(x,y-1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize C4: C4.shape =', C4.shape, 'T4.shape =', T4.shape,
          'Pt.shape =',Pt.shape)
  #   0
  #   |
  #   T4-1
  #   |
  #   2
  #   0
  #   |
  #   C4-1 ->2
  nC4 = np.tensordot(T4,C4,((2,),(0,)))
  #   0
  #   |
  #   T4-1 ->2\
  #   |        1
  #   C4-2 ->1/
  nC4 = nC4.swapaxes(1,2).reshape(len(T4),len(Pt))


  #   0
  #   |
  #   T4\
  #   |  10=Pt-1
  #   C4/
  nC4 = nC4 @ Pt
  return nC4/lg.norm(nC4)


def renormalize_C4_left(env,x,y,verbosity=0):
  """
  Renormalize corner C4 from a left move using projector P
  CPU: 2*chi**3*D**2
  """
  #  0         0         1
  #  |         |         |
  #  C4-1    2-T3-1      P
  #                     ||
  #                      0
  C4 = env.get_C4(x,y)
  T3 = env.get_T3(x+1,y)
  P = env.get_P(x,y-1)
  if verbosity > 1:
    print('Renormalize C4: C4.shape =', C4.shape, 'T3.shape =', T3.shape,
          'P.shape =',P.shape)
  #     0     0 ->1
  #     |     |
  #     C4-12-T3-1 ->2
  nC4 = np.tensordot(C4,T3,((1,),(2,)))
  #        0
  #       /  \
  #     0     1
  #     |     |
  #     C4----T3-2 -> 1
  nC4 = nC4.reshape(P.shape[0],T3.shape[2])
  #        1 ->0
  #        |
  #        P
  #       ||
  #        0
  #        0
  #       /  \
  #     C4----T3-1
  nC4 = P.T @ nC4
  return nC4/lg.norm(nC4)


def renormalize_T4(env,x,y,verbosity=0):
  """
  Renormalize edge T4 using projectors P and Pt
  CPU: chi**2*D**4*(2*chi+D**4)
  """
  #       1
  #       |
  #       P
  #     /   \
  #    0     0'
  #    0     0
  #    |     |
  #    T4-13-a-1
  #    |     |
  #    2     2
  #    0     0'
  #     \   /
  #       Pt
  #       |
  #       1
  P = env.get_P(x,y-1)
  T4 = env.get_T4(x,y)
  a = env.get_a(x+1,y)
  Pt = env.get_Pt(x,y)

  last_chi = T4.shape[0]
  D2 = a.shape[0]
  chi = P.shape[1]
  nT4 = np.dot(T4.reshape(D2*last_chi,last_chi),Pt.reshape(last_chi,D2*chi)).reshape(last_chi,D2,D2,chi)
  nT4 = nT4.transpose(2,1,0,3).reshape(D2**2,last_chi*chi)
  nT4 = np.dot(a.reshape(D2**2, D2**2),nT4).reshape(D2,D2,last_chi,chi)
  nT4 = nT4.transpose(2,0,1,3).reshape(P.shape[0], D2*chi)
  nT4 = np.dot(P.T,nT4).reshape(chi, D2, chi)
  nT4 = nT4/lg.norm(nT4)
  return nT4


def renormalize_C1_left(env,x,y,verbosity=0):
  """
  Renormalize corner C1 from a left move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #                      0
  #  C1-0    2-T1-0      ||
  #  |         |         Pt
  #  1         1         |
  #                      1
  C1 = env.get_C1(x,y)
  T1 = env.get_T1(x+1,y)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize C1: C1.shape =', C1.shape, 'T1.shape =', T1.shape,
          'Pt.shape =',Pt.shape)

  #  C1-02-T1-0
  #  |     |
  #  1     1
  #  2     1
  #  1     2
  #   \   /
  #     1
  nC1 = np.tensordot(T1,C1,((2,),(0,)))
  nC1 = nC1.swapaxes(1,2).reshape(T1.shape[0],len(Pt))
  #  C1--T1-0
  #  |   |
  #   \ /
  #    1
  #    0
  #    ||
  #    Pt
  #    |
  #    1
  nC1 = nC1 @ Pt
  return nC1/lg.norm(nC1)
