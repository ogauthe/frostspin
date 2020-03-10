import numpy as np
import scipy.linalg as lg


###############################################################################
#  construct 2x2 corners
###############################################################################

def construct_UR_corner(env,x,y,verbosity=0):
  #               0
  #               |
  #  0-T1-2     1-T4      0-C1
  #    |          |         |
  #    1          2         1
  a = env.get_a(x+2,y+1)
  T4 = env.get_T4(x+3,y+1)
  C1 = env.get_C1(x+3,y)
  T1 = env.get_T1(x+2,y)
  if verbosity > 1:
    print('construct_UR_corner: T4.shape =', T4.shape, 'C1.shape =', C1.shape,
          'T1.shape =',T1.shape)
  #  0-T1-20-C1
  #    |     |
  #    1     1->2
  cornerUR = np.tensordot(T1,C1,((2,),(0,)))
  #  0-T1----C1
  #    |     |
  #    1     2
  #          0
  #          |
  #     2<-1-T4
  #          |
  #          2-> 3
  cornerUR = np.tensordot(cornerUR,T4,((2,),(0,)))
  #    0-T1---C1
  #      |    |
  #      1    |
  #      0    |
  #      |    |
  # 2<-1-a-32-T4
  #      |    |
  #  3 <-2    3 -> 1
  cornerUR = np.tensordot(cornerUR,a,((1,2),(0,3)))
  #    /00-T1-C1
  #   0    |  |
  #    \12-a--T4
  #        |  |
  #        3  1
  #        3  2
  #         \/
  #         1
  cornerUR = cornerUR.swapaxes(1,2)
  cornerUR = cornerUR.reshape(T1.shape[0]*a.shape[1],T4.shape[2]*a.shape[2])
  return cornerUR

def construct_UL_corner(env,x,y,verbosity=0):
  #                     0
  #                     |
  #  0-T1-2    C2-1     T2-2
  #    |       |        |
  #    1       0        1
  a = env.get_a(x+1,y+1)
  T1 = env.get_T1(x+1,y)
  C2 = env.get_C2(x,y)
  T2 = env.get_T2(x+2,y)
  if verbosity > 1:
    print('construct_UL_corner: T1.shape =', T1.shape, 'C2.shape =', C2.shape,
          'T2.shape =',T2.shape)

  #  C2-10-T1-2->1
  #  |     |
  #  0->2  1->0
  cornerUL = np.tensordot(T1,C2,((0,),(1,)))

  #  C2---T1-1
  #  |    |
  #  2    0
  #  0
  #  |
  #  T2-2 -> 3
  #  |
  #  1 -> 2
  cornerUL = np.tensordot(cornerUL,T2,((2,),(0,)))

  #  C2----T1-1 -> 0
  #  |     |
  #  |     0
  #  |     0
  #  |     |
  #  T2-31-a-3
  #  |     |
  #  2->1  2
  cornerUL = np.tensordot(cornerUL,a,((0,3),(0,1)))

  #  C2-T1-0->2\
  #  |  |       1
  #  T2-a-3->3 /
  #  |  |
  #  1  2
  #  0  1
  #  \ /
  #   0
  cornerUL = cornerUL.transpose(1,2,0,3)
  cornerUL = cornerUL.reshape(T2.shape[1]*a.shape[2],T1.shape[2]*a.shape[3])
  return cornerUL


def construct_DL_corner(env,x,y,verbosity=0):
  #    0           0             0
  #    |           |             |
  #    T2-2      1-T3-2          C3-1
  #    |
  #    1
  a = env.get_a(x+1,y+2)
  T2 = env.get_T2(x,y+2)
  C3 = env.get_C3(x,y+3)
  T3 = env.get_T3(x+1,y+3)
  if verbosity > 1:
    print('construct_DL_corner: T2.shape =', T2.shape, 'C3.shape =', C3.shape,
          'T3.shape =',T3.shape)

  #      0
  #      |
  #      T2-2 -> 1
  #      |
  #      1
  #      0
  #      |
  #      C3-0 -> 2
  cornerDL = np.tensordot(T2,C3,((1,),(0,)))

  #      0
  #      |
  #      T2-1
  #      |
  #      |
  #      |     0 -> 2
  #      |     |
  #      C3-21-T3-2 -> 3
  cornerDL = np.tensordot(cornerDL,T3,((2,),(1,)))

  #      0     0 -> 2
  #      |     |
  #      T2-11-a--3
  #      |     |
  #      |     2
  #      |     2
  #      |     |
  #      C3----T3-3 -> 1
  cornerDL = np.tensordot(cornerDL,a,((1,2),(1,2)))

  #       0
  #       /\
  #      0  1
  #      0  2
  #      |  |
  #      T2-a--33\
  #      |  |     1
  #      C3-T3-12/
  cornerDL = cornerDL.swapaxes(1,2)
  cornerDL = cornerDL.reshape(T2.shape[0]*a.shape[0],T3.shape[2]*a.shape[3])
  return cornerDL


def construct_DR_corner(env,x,y,verbosity=0):
  #    0           0             0
  #    |           |             |
  #  1-T4        1-T3-2        1-C4
  #    |
  #    2
  a = env.get_a(x+2,y+2)
  T3 = env.get_T3(x+2,y+3)
  C4 = env.get_C4(x+3,y+3)
  T4 = env.get_T4(x+3,y+2)
  if verbosity > 1:
    print('construct_DR_corner: T3.shape =', T3.shape, 'C4.shape =', C4.shape,
          'T4.shape =',T4.shape)
  #       0     0->2
  #       |     |
  #     1-T3-21-C4
  cornerDR = np.tensordot(T3,C4,((2,),(1,)))

  #             0->2
  #             |
  #       3 <-1-T4
  #             |
  #             2
  #       0     2
  #       |     |
  #     1-T3----C4
  cornerDR = np.tensordot(cornerDR,T4,((2,),(2,)))

  #       0    2->3
  #       |    |
  #     1-a-33-T4
  #       |    |
  #       2    |
  #       0    |
  #       |    |
  #  2<-1-T3---C4
  cornerDR = np.tensordot(a,cornerDR,((2,3),(0,3)))

  #        0
  #        /\
  #       1  0
  #       0  3
  #       |  |
  #   /31-a--T4
  #  1    |  |
  #   \22-T3-C4
  cornerDR = cornerDR.transpose(3,0,2,1)
  cornerDR = cornerDR.reshape(T4.shape[0]*a.shape[0],T3.shape[1]*a.shape[1])
  return cornerDR



###############################################################################
# construct halves from corners
###############################################################################

def construct_U_half(env,x,y,verbosity=0):
  cornerUR = construct_UR_corner(env,x,y,verbosity)
  cornerUL = construct_UL_corner(env,x,y,verbosity)
  #  UL-10-UR
  #  |     |
  #  0     1
  return cornerUL @ cornerUR


def construct_L_half(env,x,y,verbosity=0):
  cornerUL = construct_UL_corner(env,x,y,verbosity)
  cornerDL = construct_DL_corner(env,x,y,verbosity)
  # UL-1
  # |
  # 0
  # 0
  # |
  # DL-1 ->0
  return cornerDL.T @ cornerUL


def construct_D_half(env,x,y,verbosity=0):
  cornerDL = construct_DL_corner(env,x,y,verbosity)
  cornerDR = construct_DR_corner(env,x,y,verbosity)
  #  1<- 0     0
  #      |     |
  #      DL-11-DR
  return cornerDR @ cornerDL.T


def construct_R_half(env,x,y,verbosity=0):
  cornerDR = construct_DR_corner(env,x,y,verbosity)
  cornerUR = construct_UR_corner(env,x,y,verbosity)
  #   0-UR
  #     |
  #     1
  #     0
  #     |
  #   1-DR
  return cornerUR @ cornerDR

