import numpy as np
import scipy.linalg as lg


###############################################################################
#  construct 2x2 corners
###############################################################################

def construct_UL_corner(env,x,y,verbosity=0):
  #                     0
  #                     |
  #  2-T1-0    C1-0     T4-1
  #    |       |        |
  #    1       1        2
  T4 = env.get_T4(x,y+1)
  C1 = env.get_C1(x,y)
  T1 = env.get_T1(x+1,y)
  a = env.get_a(x+1,y+1)
  if verbosity > 1:
    print('construct_UL_corner: T4.shape =', T4.shape, 'C1.shape =', C1.shape,
          'T1.shape =',T1.shape)

  #  C1-02-T1-0
  #  |     |
  #  1->2  1
  cornerUL = np.tensordot(T1,C1,((2,),(0,)))

  #  C1---T1-0
  #  |     |
  #  2     1
  #  0
  #  |
  #  T4-1 -> 2
  #  |
  #  2 -> 3
  cornerUL = np.tensordot(cornerUL,T4,((2,),(0,)))

  #  C1----T1-0
  #  |     |
  #  |     1
  #  |     0
  #  |     |
  #  T4-23-a-1->2
  #  |     |
  #  3->1  2->3
  cornerUL = np.tensordot(cornerUL,a,((1,2),(0,3)))

  #  C1-T1-0---\
  #  |  |       0
  #  T4-a-2->1 /
  #  |  |
  #  1  3
  #  2  3
  #  \ /
  #   1
  cornerUL = cornerUL.swapaxes(1,2).reshape(T1.shape[0]*a.shape[1],T4.shape[2]*a.shape[2])
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

def construct_U_half(env,x,y,verbosity=0):
  cornerUL = construct_UL_corner(env,x,y,verbosity)
  cornerUR = construct_UR_corner(env,x,y,verbosity)
  #  UL-01-UR
  #  |      |
  #  1      0
  return cornerUR @ cornerUL


def construct_L_half(env,x,y,verbosity=0):
  cornerUL = construct_UL_corner(env,x,y,verbosity)
  cornerDL = construct_DL_corner(env,x,y,verbosity)
  #  UL-0
  #  |
  #  1
  #  0
  #  |
  #  DL-1
  return cornerUL @ cornerDL


def construct_D_half(env,x,y,verbosity=0):
  cornerDL = construct_DL_corner(env,x,y,verbosity)
  cornerDR = construct_DR_corner(env,x,y,verbosity)
  #  0      1
  #  0      0
  #  |      |
  #  DL-11-DR
  return cornerDL @ cornerDR.T


def construct_R_half(env,x,y,verbosity=0):
  cornerDR = construct_DR_corner(env,x,y,verbosity)
  cornerUR = construct_UR_corner(env,x,y,verbosity)
  #      1-UR
  #         |
  #         0
  #         0
  #         |
  #  0<- 1-DR
  return cornerDR.T @ cornerUR

