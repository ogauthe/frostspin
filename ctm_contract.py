import numpy as np


###############################################################################
#  construct 2x2 corners
###############################################################################

def contract_ul_corner(C1,T1,T4,A):
  #  C1-03-T1-0
  #  |     ||
  #  1->3  12
  ul = np.dot(T1.reshape(T1.shape[0]*T1.shape[1]**2,T1.shape[3]),C1)

  #  C1---T1-0
  #  |    ||
  #  2    12
  #  0
  #  |
  #  T4=1,2 -> 3,4
  #  |
  #  3 -> 5
  ul = np.dot(ul,T4.reshape(T4.shape[0],T4.shape[1]**2*T4.shape[3])).reshape(
       T1.shape[0],T1.shape[1],T1.shape[2],T4.shape[1],T4.shape[2],T4.shape[3])
  ul = ul.transpose(0,5,2,4,1,3).reshape(T1.shape[0]*T4.shape[3]*T1.shape[2]*T4.shape[2], T1.shape[1]*T4.shape[1])
  temp = A.transpose(2,5,0,1,3,4).reshape(A.shape[2]*A.shape[5], A.shape[0]*A.shape[1]*A.shape[3]*A.shape[4])

  #  C1----T1-0
  #  |     ||
  #  |     42
  #  |   2 0
  #  |    \|
  #  T4-51-A-4
  #  | \3  |\3
  #  1     5
  ul = np.dot(ul,temp).reshape(T1.shape[0],T4.shape[3],T1.shape[2],T4.shape[2],A.shape[0],A.shape[1],A.shape[3],A.shape[4])
  ul = ul.transpose(0,1,6,7,2,3,4,5).reshape(T1.shape[0]*T4.shape[3]*A.shape[3]*A.shape[4],
                                                       T1.shape[2]*T4.shape[2]*A.shape[0]*A.shape[1])
  #  C1----T1-0
  #  |     ||
  #  |     |4
  #  |   6 |        2 0
  #  |    \|         \|
  #  T4----A-2      1-A*-4
  #  | \5  |\7        |\
  #  1     3          5 3
  ul = np.dot(ul,temp.reshape(ul.shape[1],A.shape[3]*A.shape[4]).conj()).reshape(T1.shape[0],T4.shape[3],A.shape[3],A.shape[4], A.shape[3],A.shape[4])

  #  C1-T1-0--------\
  #  |  ||           0
  #  T4=AA=2,4->1,2-/
  #  |  ||
  #  1  35
  #  3  45
  #  \ /
  #   1
  ul = ul.transpose(0,2,4,1,3,5).reshape(T1.shape[0]*A.shape[3]**2, T4.shape[3]*A.shape[4]**2)
  return ul


def contract_ur_corner(T1,C2,A,T2):
  #  3-T1-01-C2
  #    ||     |
  #    12     0
  ur = np.dot(C2,T1.reshape(T1.shape[0],T1.shape[1]*T1.shape[2]*T1.shape[3]))

  #     2-T1---C1
  #       ||    |
  #       01    3
  #             0
  #             |
  #   4,5<-2,3=T2
  #             |
  #         3<- 1
  ur = np.dot(ur.T,T2.reshape((T2.shape[0],T2.shape[1]*T2.shape[2]*T2.shape[3]))).reshape(T1.shape[1],T1.shape[2],
        T1.shape[3],T2.shape[1],T2.shape[2],T2.shape[3])
  ur = ur.transpose(3,2,1,5,0,4).reshape(T2.shape[1]*T1.shape[3]*A.shape[2]*A.shape[3], A.shape[2]*A.shape[3])
  temp = A.transpose(2,3,0,1,4,5).reshape(ur.shape[1], A.shape[0]*A.shape[1]*A.shape[4]*A.shape[5])
  #     1-T1---C1
  #       ||    |
  #       42    |
  #     2 0     |
  #      \|     |
  #     5-A-15-T2
  #       |\ 3  |
  #       4 3   0
  ur = np.dot(ur,temp).reshape(T2.shape[1]*T1.shape[3],A.shape[2]*A.shape[3]*A.shape[0]*A.shape[1],A.shape[4]*A.shape[5])
  ur = ur.swapaxes(1,2).reshape(T2.shape[1]*T1.shape[3]*A.shape[4]*A.shape[5],A.shape[2]*A.shape[3]*A.shape[0]*A.shape[1])
  ur = np.dot(ur,temp.reshape(ur.shape[1],A.shape[4]*A.shape[5]).conj()).reshape(T2.shape[1],T1.shape[3],A.shape[4],A.shape[5],A.shape[4],A.shape[5])
  #    /31-T1-C1
  #   1    ||  |
  #    \43=AA=T2
  #     55 ||  |
  #        24  0
  #        12  0
  #         \ /
  #          0
  ur = ur.transpose(0,2,4,1,3,5).reshape(T2.shape[1]*A.shape[4]**2,T1.shape[3]*A.shape[5]**2)
  return ur



def contract_dr_corner(A,T2,T3,C3):
  #         12     0
  #         ||     |
  #       3-T3-01-C3
  dr = T3.transpose(2,0,1,3).reshape(T3.shape[2],T3.shape[0]*T3.shape[1]*T3.shape[3])
  dr = np.dot(C3,dr)

  #              0
  #              |
  #         1,2=T2
  #              |
  #              3
  #       12     0
  #       ||     |
  #     3-T3----C3
  temp = T2.transpose(0,2,3,1).reshape(T2.shape[0]*T2.shape[2]*T2.shape[3],T2.shape[1])
  dr = np.dot(temp,dr).reshape(T2.shape[0],T2.shape[2],T2.shape[3],T3.shape[0],T3.shape[1],T3.shape[3])
  dr = dr.transpose(0,5,2,4,1,3).reshape(T2.shape[0]*T3.shape[3]*T2.shape[3]*T3.shape[1],T2.shape[2]*T3.shape[0])
  temp = A.transpose(3,4,0,1,2,5).reshape(dr.shape[1], A.shape[0]*A.shape[1]*A.shape[2]*A.shape[5])
  dr = np.dot(dr,temp).reshape(T2.shape[0]*T3.shape[3],T2.shape[3]*T3.shape[1]*A.shape[0]*A.shape[1],A.shape[2]*A.shape[5])
  #      2 4     0
  #       \|     |
  #      5-A-02=T2
  #        |\ 2  |
  #        1 3   |
  #        53    |
  #        ||    |
  #      1-T3---C3
  dr = dr.swapaxes(1,2).reshape(T2.shape[0]*T3.shape[3]*A.shape[2]*A.shape[5],T2.shape[3]*T3.shape[1]*A.shape[0]*A.shape[1])
  dr = np.dot(dr,temp.reshape(dr.shape[1],A.shape[2]*A.shape[5]).conj()).reshape(T2.shape[0],T3.shape[3],A.shape[2],A.shape[5],A.shape[2],A.shape[5])

  #           0
  #          / \
  #         12  0
  #         24  0
  #         ||  |
  #     /43-AA=T2
  #    1 55 ||  |
  #     \31-T3-C3
  dr = dr.transpose(0,2,4,1,3,5).reshape(T2.shape[0]*A.shape[2]**2, T3.shape[3]*A.shape[5]**2)
  return dr


def contract_dl_corner(T4,A,C4,T3):
  #      0
  #      |
  #      T4=1,2
  #      |
  #      3
  #      0
  #      |
  #      C4-1
  dl = np.dot(T4.reshape(T4.shape[0]*T4.shape[1]*T4.shape[2],T4.shape[3]),C4)
  #      0
  #      |
  #      T4=1,2
  #      |     34
  #      |     01
  #      |     ||
  #      C4-33-T3-25
  dl = np.dot(dl,T3.reshape(T3.shape[0]*T3.shape[1]*T3.shape[2],T3.shape[3]).T).reshape(T4.shape[0],T4.shape[1],T4.shape[2],T3.shape[0],T3.shape[1],T3.shape[2])

  temp = A.transpose(4,5,0,1,2,3).reshape(A.shape[4]*A.shape[5],A.shape[0]*A.shape[1]*A.shape[2]*A.shape[3])
  dl = dl.transpose(0,5,4,2,3,1).reshape(T4.shape[0]*T3.shape[2]*temp.shape[0],temp.shape[0])
  #      0   2 4
  #      |    \|
  #      T4=51-A--5
  #      |  3  |\
  #      |     0 3
  #      |     42
  #      |     ||
  #      C4----T3-1
  dl = np.dot(dl,temp).reshape(T4.shape[0]*T3.shape[2],temp.shape[0]*A.shape[0]*A.shape[1],A.shape[2]*A.shape[3])
  dl = dl.swapaxes(1,2).reshape(dl.shape[0]*dl.shape[2],dl.shape[1])
  dl = np.dot(dl, temp.reshape(dl.shape[1],A.shape[2]*A.shape[3]).conj()).reshape(T4.shape[0],T3.shape[2],A.shape[2],A.shape[3],A.shape[2],A.shape[3])
  #        0
  #       / \
  #      0   12
  #      0   24
  #      |   ||
  #      T4==AA=34\
  #      |   || 55 1
  #      C3--T3-13/
  dl = dl.transpose(0,2,4,1,3,5).reshape(T4.shape[0]*A.shape[2]**2,T3.shape[2]*A.shape[3]**2)
  return dl



###############################################################################
# construct halves from corners
###############################################################################

def contract_u_half(C1,T1l,T1r,C2,T4,Al,Ar,T2):
  ul = contract_ul_corner(C1,T1l,T4,Al)
  ur = contract_ur_corner(T1r,C2,Ar,T2)
  #  UL-01-UR
  #  |      |
  #  1      0
  return ur @ ul


def contract_l_half(C1,T1,T4u,Au,T4d,Ad,C4,T3):
  ul = contract_ul_corner(C1,T1,T4u,Au)
  dl = contract_dl_corner(T4d,Ad,C4,T3)
  #  UL-0
  #  |
  #  1
  #  0
  #  |
  #  DL-1
  return ul @ dl


def contract_d_half(T4,Al,Ar,T2,C4,T3l,T3r,C3):
  dl = contract_dl_corner(T4,Al,C4,T3l)
  dr = contract_dr_corner(Ar,T2,T3r,C3)
  #  0      1
  #  0      0
  #  |      |
  #  DL-11-DR
  return dl @ dr.T


def contract_r_half(T1,C2,Au,T2u,Ad,T2d,T3,C3):
  ur = contract_ur_corner(T1,C2,Au,T2u)
  dr = contract_dr_corner(Ad,T2d,T3,C3)
  #      1-UR
  #         |
  #         0
  #         0
  #         |
  #  0<- 1-DR
  return dr.T @ ur

