import numpy as np
import scipy.linalg as lg


def construct_projectors(R,Rt,chi,verbosity=0):
  if verbosity > 1:
    print('construct projectors: R.shape =',R.shape, 'Rt.shape =', Rt.shape)
  # convention : for every move, leg 0 of R and Rt are to be contracted
  U,s,V = lg.svd(R.T @ Rt)
  s12 = 1/np.sqrt(s[:chi])
  # convention: double-leg has index 0, single leg has index 1
  #  0        1
  #  ||       |
  #  Pt       P
  #  |        ||
  #  1        0
  Pt = Rt @ np.einsum('ij,i->ji', V[:chi].conj(), s12)
  P = R @ np.einsum('ij,j->ij', U[:,:chi].conj(), s12)
  return P,Pt


def renormalize_C1_up(env,x,y,verbosity=0):
  """
  Renormalize corner C1 from an up move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #                0
  #                |
  #       0-C1   1-T4    1-Pt=0
  #         |      |
  #         1      2
  C1 = env.get_C1(x,y)
  T4 = env.get_T4(x,y+1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
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
  nC1 = np.tensordot(C1,T4,(1,0)).reshape(Pt.shape[0],T4.shape[2])
  #             /-C1
  #  0<- 1-Pt=00  |
  #             \-T4
  #               |
  #               2 -> 1
  nC1 = Pt.T @ nC1
  return nC1/lg.norm(nC1)


def renormalize_T1(env,x,y,verbosity=0):
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
  T1 = env.get_T1(x,y)
  a = env.get_a(x,y+1)
  P = env.get_P(x+1,y)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize T1: T1.shape =', T1.shape, 'P.shape =', P.shape,
          'Pt.shape =',Pt.shape)
  #     0\       # reshape P to 3-leg tensor
  #       P-2
  #     1/
  P3 = P.reshape(T1.shape[2],a.shape[3],P.shape[1])
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
  #    /---0-T1\
  #   0      |  P-1 ->3 ->2
  #    \1<-2-a-/
  #          |
  #          3 ->2 ->1
  nT1 = nT1.transpose(0,2,3,1).reshape(Pt.shape[0],a.shape[2],Pt.shape[1])
  #             /T1\
  #  0<- 1-Pt=00 |  P-2
  #             \a-/
  #              |
  #              1
  nT1 = np.tensordot(Pt,nT1,((0,),(0,)))
  return nT1/lg.norm(nT1)


def renormalize_C2_up(env,x,y,verbosity=0):
  """
  Renormalize corner C2 from an up move using projector P
  CPU: 2*chi**3*D**2
  """
  #            0
  #            |
  #  C2-1      T2-2    0=P-1
  #  |         |
  #  0         1
  C2 = env.get_C2(x,y)
  T2 = env.get_T2(x,y+1)
  P = env.get_Pt(x+1,y)
  if verbosity > 1:
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


def renormalize_C2_left(env,x,y,verbosity=0):
  """
  Renormalize corner C2 from a left move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #                      0
  #  C2-1    0-T1-2      ||
  #  |         |         Pt
  #  0         1         |
  #                      1
  C2 = env.get_C2(x,y)
  T1 = env.get_T1(x+1,y)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
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


def renormalize_T2(env,x,y,verbosity=0):
  """
  Renormalize edge T2 using projectors P and Pt
  CPU: chi**2*D**4*(2*chi+D**4)
  """
  #    0          0        0         1
  #    |          ||       |         |
  #    T2-2       Pt     1-a-3       P
  #    |          |        |         ||
  #    1          1        2         0
  T2 = env.get_T2(x,y)
  a = env.get_a(x+1,y)
  P = env.get_P(x,y-1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
    print('Renormalize T2: T2.shape =', T2.shape, 'P.shape =', P.shape,
          'Pt.shape =',Pt.shape)

  #       2    # reshape P to 3-leg tensor
  #       |
  #       P3
  #     /   \
  #     0    1
  P3 = P.reshape(T2.shape[0],a.shape[0],P.shape[1])
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
  nT2 = nT2.reshape(P.shape[1],len(Pt),a.shape[2])

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


def renormalize_C3_left(env,x,y,verbosity=0):
  """
  Renormalize corner C3 from a left move using projector P
  CPU: 2*chi**3*D**2
  """
  #  0         0         1
  #  |         |         |
  #  C3-1    1-T3-2      P
  #                     ||
  #                      0
  C3 = env.get_C3(x,y)
  T3 = env.get_T3(x+1,y)
  P = env.get_P(x,y-1)
  if verbosity > 1:
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
  nC3 = nC3.reshape(P.shape[0],T3.shape[2])

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


def renormalize_C3_down(env,x,y,verbosity=0):
  """
  Renormalize corner C3 from a down move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #            0
  #            |
  #  0         T2-2
  #  |         |         0=Pt-1
  #  C3-1      1
  C3 = env.get_C3(x,y)
  T2 = env.get_T2(x,y-1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
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


def renormalize_T3(env,x,y,verbosity=0):
  """
  Renormalize edge T3 using projectors P and Pt
  CPU: chi**2*D**4*(2*chi+D**4)
  """
  #                    0
  #                    |
  #     0   1-P=0    1-a-3    0=Pt-1
  #     |              |
  #   1-T3-2           2
  T3 = env.get_T3(x,y)
  a = env.get_a(x,y-1)
  P = env.get_P(x-1,y)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
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


def renormalize_C4_down(env,x,y,verbosity=0):
  """
  Renormalize corner C4 from down move using projector P
  CPU: 2*chi**3*D**2
  """
  #    0         0
  #    |         |
  #  1-C4      1-T4      1-P=0
  #              |
  #              2
  C4 = env.get_C4(x,y)
  T4 = env.get_T4(x,y-1)
  P = env.get_P(x-1,y)
  if verbosity > 1:
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


def renormalize_C4_right(env,x,y,verbosity=0):
  """
  Renormalize corner C4 from right move using projector Pt
  CPU: 2*chi**3*D**2
  """
  #    0         0            1
  #    |         |            |
  #  1-C4      1-T3-2         Pt
  #                           ||
  #                           0
  C4 = env.get_C4(x,y)
  T3 = env.get_T3(x-1,y)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
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


def renormalize_T4(env,x,y,verbosity=0):
  """
  Renormalize edge T4 using projectors P and Pt
  CPU: chi**2*D**4*(2*chi+D**4)
  """
  #  1       0         0       0
  #  |       |         |       ||
  #  Pt    1-T4      1-a-3     P
  #  ||      |         |       |
  #  0       2         2       1
  T4 = env.get_T4(x,y)
  a = env.get_a(x-1,y)
  P = env.get_P(x,y+1)
  Pt = env.get_Pt(x,y)
  if verbosity > 1:
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
  nT4 = nT4.swapaxes(1,2).reshape(Pt.shape[1],a.shape[1],P.shape[0])
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


def renormalize_C1_right(env,x,y,verbosity=0):
  """
  Renormalize corner C1 from right move using projector P
  CPU: 2*chi**3*D**2
  """
  #                      0
  #                      ||
  #  0-C1     0-T1-2     P
  #    |        |        |
  #    1        1        1
  C1 = env.get_C1(x,y)
  T1 = env.get_T1(x-1,y)
  P = env.get_P(x,y+1)
  if verbosity > 1:
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
  nC1 = nC1.swapaxes(0,2).reshape(P.shape[0],T1.shape[0])
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

