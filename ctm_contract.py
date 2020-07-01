import numpy as np


###############################################################################
#  construct 2x2 corners
###############################################################################

def construct_UL_corner(C1,T1,T4,A):
  #  C1-03-T1-0
  #  |     ||
  #  1->2  12
  cornerUL = np.dot(T1.reshape(T1.shape[0]*T1.shape[1]**2),C1).reshape(T1.shape[0]*T1.shape[1]**2,C1.shape[1])

  #  C1---T1-0
  #  |    ||
  #  2    12
  #  0
  #  |
  #  T4=1,2 -> 3,4
  #  |
  #  3 -> 5
  cornerUL = np.dot(cornerUL,T4.reshape(T4.shape[1]**2*T4.shape[3])).reshape(
       T1.shape[0],T1.shape[1],T1.shape[2],T4.shape[1],T4.shape[2],T4.shape[3])
  cornerUL = cornerUL.transpose(0,5,2,4,1,3).reshape(T1.shape[0]*T4.shape[3]*T1.shape[2]*T4.shape[2], T1.shape[1]*T4.shape[1])
  temp = A.transpose(2,5,0,1,3,4).reshape(A.shape[2]*A.shape[5], A.shape[0]*A.shape[1]*A.shape[3]*A.shape[4])

  #  C1----T1-0
  #  |     ||
  #  |     42
  #  |   2 0
  #  |    \|
  #  T4-51-A-4
  #  | \3  |\3
  #  1     5
  cornerUL = np.dot(cornerUL,temp).reshape(T1.shape[0],T4.shape[0],T1.shape[2],T4.shape[2],A.shape[0],A.shape[1],A.shape[3],A.shape[4])
  cornerUL = cornerUL.transpose(0,1,6,7,2,3,4,5).reshape(T1.shape[0]*T4.shape[0]*A.shape[3]*A.shape[4],
                                                       T1.shape[2]*T4.shape[2]*A.shape[0]*A.shape[1])
  #  C1----T1-0
  #  |     ||
  #  |     |4
  #  |   6 |        2 0
  #  |    \|         \|
  #  T4----A-2      1-A*-4
  #  | \5  |\7        |\
  #  1     3          5 3
  cornerUL = np.dot(cornerUL,temp.reshape(cornerUL.shap[1],A.shape[3]*A.shape[4].conj()).reshape(T1.shape[0],T4.shape[0],A.shape[3],A.shape[4], A.shape[3],A.shape[4])

  #  C1-T1-0--------\
  #  |  ||           0
  #  T4=AA=2,4->1,2-/
  #  |  ||
  #  1  35
  #  3  45
  #  \ /
  #   1
  cornerUL = cornerUL.transpose(0,2,4,1,3,5).reshape(T1.shape[0]*A.shape[3]**2, T4.shape[2]*A.shape[4]**2)
  return cornerUL


def construct_UR_corner(env,x,y,verbosity=0):
  #                      0
  #                      |
  #  2-T1-0   1-C2    2-T2
  #    |         |       |
  #    1         0       1
  T1 = env.get_T1(x+2,y)
  C2 = env.get_C2(x+3,y)
  T2 = env.get_T2(x+3,y+1)
  a = env.get_a(x+2,y+1)
  if verbosity > 1:
    print('construct_UR_corner: T1.shape =', T1.shape, 'C2.shape =', C2.shape,
          'T2.shape =',T2.shape)
  #  2-T1-01-C2
  #    |      |
  #    1      0
  cornerUR = np.tensordot(C2,T1,((1,),(0,)))

  # 1<- 2-T1---C1
  #       |     |
  #       1->0  0
  #             0
  #             |
  #       3<-2-T2
  #             |
  #             1-> 2
  cornerUR = np.tensordot(cornerUR,T2,((0,),(0,)))
  # 0<- 1-T1---C1
  #       |     |
  #       0     |
  #       0     |
  #       |     |
  #     3-a-13-T2
  #       |     |
  #       2     2 -> 1
  cornerUR = np.tensordot(cornerUR,a,((0,3),(0,1)))
  #    /20-T1-C1
  #   1    |   |
  #    \33-a--T2
  #        |   |
  #        2   1
  #        1   0
  #         \ /
  #          0
  cornerUR = cornerUR.transpose(1,2,0,3).reshape(T2.shape[1]*a.shape[2],T1.shape[2]*a.shape[3])
  return cornerUR



def construct_DR_corner(env,x,y,verbosity=0):
  #     0       0       0
  #     |       |       |
  #  2-T2    1-C3     2-T3-1
  #     |
  #     1
  T2 = env.get_T2(x+3,y+2)
  C3 = env.get_C3(x+3,y+3)
  T3 = env.get_T3(x+2,y+3)
  a = env.get_a(x+2,y+2)
  if verbosity > 1:
    print('construct_DR_corner: T2.shape =', T2.shape, 'C3.shape =', C3.shape,
          'T3.shape =',T3.shape)

  #         0      0->2
  #         |      |
  #   1<- 2-T3-11-C3
  cornerDR = np.tensordot(T3,C3,((1,),(1,)))

  #              0
  #              |
  #       1 <-2-T2
  #              |
  #              1
  #       0->2   2
  #       |      |
  #  3<-1-T3----C3
  cornerDR = np.tensordot(T2,cornerDR,((1,),(2,)))

  #    2<- 0     0
  #        |     |
  #      3-a-11-T2
  #        |     |
  #        2     |
  #        2     |
  #        |     |
  #  1<- 3-T3---C3
  cornerDR = np.tensordot(cornerDR,a,((1,2),(1,2)))

  #           0
  #          / \
  #         1   0
  #         2   0
  #         |   |
  #     /33-a--T2
  #    1    |   |
  #     \21-T3-C3
  cornerDR = cornerDR.swapaxes(1,2).reshape(T2.shape[0]*a.shape[0],T3.shape[2]*a.shape[3])
  return cornerDR


def construct_DL_corner(env,x,y,verbosity=0):
  #    0      0        0
  #    |      |        |
  #  2-T3-1   C4-1     T4-1
  #                    |
  #                    2
  T3 = env.get_T3(x+1,y+3)
  C4 = env.get_C4(x,y+3)
  T4 = env.get_T4(x,y+2)
  a = env.get_a(x+1,y+2)
  if verbosity > 1:
    print('construct_DL_corner: T3.shape =', T3.shape, 'C4.shape =', C4.shape,
          'T4.shape =',T4.shape)

  #      0
  #      |
  #      T4-1
  #      |
  #      2
  #      0
  #      |
  #      C4-1 -> 2
  cornerDL = np.tensordot(T4,C4,((2,),(0,)))

  #      0
  #      |
  #      T4-1
  #      |
  #      |
  #      |     0 -> 2
  #      |     |
  #      C4-22-T3-1 -> 3
  cornerDL = np.tensordot(cornerDL,T3,((2,),(2,)))

  #      0->2  0
  #      |     |
  #      T4-13-a--1
  #      |     |
  #      |     2
  #      |     2
  #      |     |
  #      C4----T3-3
  cornerDL = np.tensordot(a,cornerDL,((2,3),(2,1)))

  #        0
  #       / \
  #      0   1
  #      2   0
  #      |   |
  #      T4--a--13\
  #      |   |     1
  #      C3--T3-32/
  cornerDL = cornerDL.transpose(2,0,3,1).reshape(T4.shape[0]*a.shape[0],T3.shape[1]*a.shape[1])
  return cornerDL



###############################################################################
# construct halves from corners
###############################################################################

def construct_U_half(C1,T1l,T1r,C2,T4,Al,Ar,T2):
  cornerUL = construct_UL_corner(C1,T1l,T4,Al)
  cornerUR = construct_UR_corner(T1r,C2,Ar,T2)
  #  UL-01-UR
  #  |      |
  #  1      0
  return cornerUR @ cornerUL


def construct_L_half(C1,T1,T4u,Au,T4d,Ad,C4,T3):
  cornerUL = construct_UL_corner(C1,T1,T4u,Au)
  cornerDL = construct_DL_corner(T4d,Ad,C4,T3)
  #  UL-0
  #  |
  #  1
  #  0
  #  |
  #  DL-1
  return cornerUL @ cornerDL


def construct_D_half(T4,Al,Ar,T2,C4,T3l,T3r,C3):
  cornerDL = construct_DL_corner(T4,Al,C4,T3l)
  cornerDR = construct_DR_corner(Ar,T2,T3r,C3)
  #  0      1
  #  0      0
  #  |      |
  #  DL-11-DR
  return cornerDL @ cornerDR.T


def construct_R_half(T1,C2,Au,T2u,Ad,T2d,T3,C3)
  cornerDR = construct_DR_corner(Ad,T2d,T3,C3)
  cornerUR = construct_UR_corner(T1,C2,Au,T2u)
  #      1-UR
  #         |
  #         0
  #         0
  #         |
  #  0<- 1-DR
  return cornerDR.T @ cornerUR

