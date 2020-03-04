import numpy as np
import scipy.linalg as lg


###############################################################################
#  construct 2x2 corners
###############################################################################
def construct_UL_corner(env,x,y):
  #                     0
  #                     |
  #  0-T1-2    C1-1     T2-2
  #    |       |        |
  #    1       0        1
  a = env.get_a(x+1,y+1)
  T1 = env.get_T1(x+1,y)
  C1 = env.get_C1(x,y)
  T2 = env.get_T2(x+2,y)

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


def construct_DL_corner(env,x,y):
  #    0           0             0
  #    |           |             |
  #    T2-2      1-T3-2          C2-1
  #    |
  #    1
  a = env.get_a(x+1,y+2)
  T2 = env.get_T2(x,y+2)
  C2 = env.get_C2(x,y+3)
  T3 = env.get_T3(x+1,y+3)

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


def construct_DR_corner(env,x,y):
  #    0           0             0
  #    |           |             |
  #  1-T4        1-T3-2        1-C3
  #    |
  #    2
  a = env.get_a(x+2,y+2)
  T3 = env.get_T3(x+2,y+3)
  C3 = env.get_C3(x+3,y+3)
  T4 = env.get_T4(x+3,y+2)

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


def construct_UR_corner(env,x,y):
  #               0
  #               |
  #  0-T1-2     1-T4      0-C4
  #    |          |         |
  #    1          2         1
  a = env.get_a(x+2,y+1)
  T4 = env.get_T4(x+3,y+1)
  C4 = env.get_C4(x+3,y)
  T1 = env.get_T1(x+2,y)


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
###############################################################################

def construct_U_half(env,x,y):
  cornerUL = construct_UL_corner(env,x,y)
  cornerUR = construct_UR_corner(env,x,y)
  #  UL-10-UR
  #  |     |
  #  0     1
  return cornerUL @ cornerUR


def construct_L_half(env,x,y):
  cornerUL = construct_UL_corner(env,x,y)
  cornerDL = construct_DL_corner(env,x,y)
  # UL-1
  # |
  # 0
  # 0
  # |
  # DL-1 ->0
  return cornerDL.T @ cornerUL


def construct_D_half(env,x,y):
  cornerDL = construct_DL_corner(env,x,y)
  cornerDR = construct_DR_corner(env,x,y)
  #  1     0
  #  0     1
  #  |     |
  #  DL-10-DR
  return (cornerDL @ cornerDR).T


def construct_R_half(env,x,y):
  cornerDR = construct_DR_corner(env,x,y)
  cornerUR = construct_UR_corner(env,x,y)
  #   0-DR
  #     |
  #     1
  #     0
  #     |
  #   1-UR
  return cornerDR @ cornerUR

