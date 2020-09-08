import numpy as np
from ctm_contract import contract_UL_corner, contract_UR_corner, contract_DL_corner, contract_DR_corner


def rdm_1x1(C1,T1,C2,T4,A,T2,C4,T3,C3):
  """
  Compute 1-site reduced density matrix from CTMRG environment tensors
  """
  #   C1-0       3-T1-0         1-C2
  #   |            ||              |
  #   1            12              0
  #         0        0
  #   0      \ 2      \ 2          0
  #   |       \|       \|          |
  #   T4-1   5-A--3   5-A*-1    2-T2
  #   | \2     |\       |\      3/ |
  #   3        4 1      4 1        1
  #
  #   0           01               0
  #   |           ||               |
  #   C4-1      3-T3-2          1-C3

  # bypassing tensordot makes code conceptually simpler and memory efficient but unreadable
  rdm = np.tensordot(T4, C1, ((0,),(1,)))
  rdm = np.tensordot(rdm, C4, ((2,),(0,)))
  rdm = np.tensordot(rdm, T1, ((2,),(3,)))
  rdm = np.tensordot(rdm, A, ((4, 0),(2, 5)))
  dr = np.tensordot(T2, C2, ((0,),(0,)))
  dr = np.tensordot(dr, C3, ((0,),(0,)))
  dr = np.tensordot(dr, T3, ((3,),(2,)))
  rdm = np.tensordot(dr, rdm, ((2, 0, 3, 5),(2, 6, 7, 1)))
  rdm = np.tensordot(rdm, A.conj(), ((3, 2, 0, 1, 5),(2, 5, 3, 4, 1)))
  return rdm/np.trace(rdm)


def rdm_1x2(C1, T1l, T1r, C2, T4, Al, Ar, T2, C4, T3l, T3r, C3):
  """
  Compute reduced density matrix for 2 sites in a row
  """
  # CPU optimal, not memory optimal

  #
  #   C1-0     3-T1-0           3-T1-0       1-C2
  #   |          ||               ||            |
  #   1          12               12            0
  #        0       0        0       0
  #   0     \ 2     \ 2      \ 2     \ 2        0
  #   |      \|      \|       \|      \|        |
  #   T4-1  5-A--3  5-A*-3   5-A--3  5-A*-3  2-T2
  #   | \2    |\      |\       |\      |\    3/ |
  #   3       4 1     4 1      4 1     4 1      1
  #
  #   0          01               01            0
  #   |          ||               ||            |
  #   C4-1     3-T3-2           3-T3-2       1-C3

  right = np.tensordot(C2, T1r, ((1,),(0,)))
  right = np.tensordot(np.tensordot(C3, T2, ((0,),(1,))), right, ((1,),(0,)))
  right = np.tensordot(right, Ar, ((3, 1),(2, 3)))
  right = np.tensordot(right, Ar.conj(), ((2, 5, 1),(2, 1, 3)))
  right = np.tensordot(right, T3r, ((3, 6, 0),(0, 1, 2)))
  right = right.transpose(0,2,4,5,1,3).copy()
  left = np.tensordot(C4, T4, ((0,),(3,)))
  left = np.tensordot(left, T3l, ((0,),(3,)))
  left = np.tensordot(left, Al, ((1, 3),(5, 4)))
  left = np.tensordot(left, Al.conj(), ((1, 5, 2),(5, 1, 4)))
  left = left.transpose(1,2,4,5,7,0,3,6).copy()
  T1lC1 = np.tensordot(T1l, C1, ((3,),(0,))).swapaxes(0,3).copy()
  left = np.tensordot(left, T1lC1, ((5, 6, 7),(0, 1, 2)))
  del T1lC1
  left = left.transpose(1,3,5,2,4,0).copy()
  rdm = np.tensordot(left, right, ((2, 3, 4, 5),(0,1,2,3)))
  rdm = rdm.swapaxes(1,2).reshape(Al.shape[0]*Ar.shape[0],Al.shape[0]*Ar.shape[0])
  rdm = rdm/np.trace(rdm)
  return rdm


def rdm_2x1(C1,T1,C2,T4u,Au,T2u,T4d,Ad,T2d,C4,T3,C3):
  """
  Compute reduced density matrix for 2 sites in a column
  """
  # contract using 1x2 with swapped tensors and legs
  return rdm_1x2(C2, T2u.transpose(1,2,3,0), T2d.transpose(1,2,3,0), C3.T, T1,
                 Au.transpose(0,1,3,4,5,2), Ad.transpose(0,1,3,4,5,2),
                 T3.transpose(2,3,0,1), C1, T4u.transpose(1,2,3,0),
                 T4d.transpose(1,2,3,0),C4.T)


def rdm_2x2(C1,T1l,T1r,C2,T4u,Aul,Aur,T2u,T4d,Adl,Adr,T2d,C4,T3l,T3r,C3):
  #
  #   C1-0     3-T1-0           3-T1-0       1-C2
  #   |          ||               ||            |
  #   1          12               12            0
  #        0       0        0       0
  #   0     \ 2     \ 2      \ 2     \ 2        0
  #   |      \|      \|       \|      \|        |
  #   T4-1  5-A--3  5-A*-3   5-A--3  5-A*-3  2-T2
  #   | \2    |\      |\       |\      |\    3/ |
  #   3       4 1     4 1      4 1     4 1      1
  #
  #        0       0        0       0
  #   0     \ 2     \ 2      \ 2     \ 2        0
  #   |      \|      \|       \|      \|        |
  #   T4-1  5-A--3  5-A*-3   5-A--3  5-A*-3  2-T2
  #   | \2    |\      |\       |\      |\    3/ |
  #   3       4 1     4 1      4 1     4 1      1
  #
  #   0          01               01            0
  #   |          ||               ||            |
  #   C4-1     3-T3-2           3-T3-2       1-C3

  # memory use: 3*chi**2*D**4*d**4

  ul = np.tensordot(T1l, C1, ((3,),(0,)))
  ul = np.tensordot(ul, T4u, ((3,),(0,)))
  ul = np.tensordot(ul, Aul, ((1, 3),(2, 5)))
  ul = np.tensordot(ul, Aul.conj(), ((1, 2, 5),(2, 5, 1)))
  dl = np.tensordot(T4d, C4, ((3,),(0,)))
  dl = np.tensordot(dl, T3l, ((3,),(3,)))
  dl = np.tensordot(dl, Adl, ((1, 3),(5, 4)))
  dl = np.tensordot(dl, Adl.conj(), ((1, 2, 5),(5, 4, 1)))
  left = np.tensordot(dl, ul, ((0, 3, 6),(1, 4, 7)))
  del ul,dl
  left = left.transpose(5,7,9,2,4,0,1,3,6,8).copy()
  ur = np.tensordot(C2, T1r, ((1,),(0,)))
  ur = np.tensordot(ur, T2u, ((0,),(0,)))
  ur = np.tensordot(ur, Aur, ((0, 4),(2, 3)))
  ur = np.tensordot(ur, Aur.conj(), ((0, 3, 5),(2, 3, 1)))
  dr = np.tensordot(T2d, C3, ((1,),(0,)))
  dr = np.tensordot(dr, T3r, ((3,),(2,)))
  dr = np.tensordot(dr, Adr, ((1, 3),(3, 4)))
  dr = np.tensordot(dr, Adr.conj(), ((1, 2, 5),(3, 4, 1)))
  right = np.tensordot(dr, ur, ((3, 6, 0),(3, 6, 1)))
  del ur,dr
  right = right.transpose(1,3,6,8,5,7,9,2,4,0).copy()  # reduce memory
  rdm = np.tensordot(right, left, ((4,5,6,7,8,9),(0,1,2,3,4,5)))
  d4 = Aul.shape[0]*Aur.shape[0]*Adl.shape[0]*Adr.shape[0]
  rdm = rdm.transpose(6, 2, 4, 0, 7, 3, 5, 1).reshape(d4,d4)

  rdm /= np.trace(rdm)
  return rdm


def rdm_diag_dr(C1,T1l,T1r,C2,T4u,Aul,Aur,T2u,T4d,Adl,Adr,T2d,C4,T3l,T3r,C3):
  """
  ┌──┬──┐
  │02│  │
  ├──┼──┤
  │  │13│
  └──┴──┘
  """
  # memory use: 3*chi**2*D**4*d**4
  ul = np.tensordot(T1l, C1, ((3,),(0,)))
  ul = np.tensordot(ul, T4u, ((3,),(0,)))
  ul = np.tensordot(ul, Aul, ((1, 3),(2, 5)))
  ul = np.tensordot(ul, Aul.conj(), ((1, 2, 5),(2, 5, 1)))
  #   ------0
  #   | 2 ||
  #   |  \||
  #   |======3,6
  #   |  /||
  #   | 5 ||
  #   1   47
  ul = ul.transpose(0,3,6,2,5,1,4,7).reshape(T1l.shape[0]*Aul.shape[3]**2*Aul.shape[0]**2, T4u.shape[3]*Aul.shape[4]**2)
  dl = contract_DL_corner(T4d,Adl,C4,T3l)
  rdm = ul @ dl
  rdm = rdm.reshape(T1l.shape[0]*Aul.shape[3]**2, Aul.shape[0]**2*dl.shape[1])
  del ul,dl
  ur = contract_UR_corner(T1r,C2,Aur,T2u)
  rdm = ur @ rdm
  rdm = rdm.reshape(ur.shape[0], Aul.shape[0]**2, T3l.shape[2]*Adl.shape[3]**2)
  del ur
  #   -----           -----
  #   |11 |   --->    |00 |
  #   |   0           |   1
  #   |--2            |--2
  rdm = rdm.swapaxes(0,1).reshape(Aul.shape[0]**2, T2u.shape[1]*Aur.shape[4]**2*T3l.shape[2]*Adl.shape[3]**2)
  dr = np.tensordot(T2d, C3, ((1,),(0,)))
  dr = np.tensordot(dr, T3r, ((3,),(2,)))
  dr = np.tensordot(dr, Adr, ((1, 3),(3, 4)))
  dr = np.tensordot(dr, Adr.conj(), ((1, 2, 5),(3, 4, 1)))
  #     36   0
  #     || 2 |
  #     ||/  |              0
  # 4,7======|   ---->      |
  #     ||\  |            22|
  #     || 5 |          1----
  #   1-------
  dr = dr.transpose(0,3,6,1,4,7,2,5).reshape(rdm.shape[1],Adr.shape[0]**2)  # max mem
  rdm = rdm @ dr
  rdm = rdm.reshape(Aul.shape[0],Aul.shape[0],Adr.shape[0],Adr.shape[0])
  rdm = rdm.swapaxes(1,2).reshape(Aul.shape[0]*Adr.shape[0], Aul.shape[0]*Adr.shape[0])
  rdm /= rdm.trace()
  return rdm


def rdm_diag_ur(C1,T1l,T1r,C2,T4u,Aul,Aur,T2u,T4d,Adl,Adr,T2d,C4,T3l,T3r,C3):
  """
  ┌──┬──┐
  │  │02│
  ├──┼──┤
  │13│  │
  └──┴──┘
  """
  # memory use: 3*chi**2*D**4*d**4
  dl = np.tensordot(T4d, C4, ((3,),(0,)))
  dl = np.tensordot(dl, T3l, ((3,),(3,)))
  dl = np.tensordot(dl, Adl, ((1, 3),(5, 4)))
  dl = np.tensordot(dl, Adl.conj(), ((1, 2, 5),(5, 4, 1)))
  #  0  36
  #  |2 ||
  #  | \||
  #  |====4,7
  #  | /||
  #  |5 ||
  #  ------1
  dl = dl.transpose(0,3,6,2,5,1,4,7).reshape(T4d.shape[0]*Adl.shape[2]**2, Adl.shape[0]**2*T3l.shape[2]*Adl.shape[3]**2)
  ul = contract_UL_corner(C1,T1l,T4u,Aul)
  rdm = ul @ dl
  rdm = rdm.reshape(ul.shape[0]*Adl.shape[0]**2, T3l.shape[2]*Adl.shape[3]**2)
  del ul,dl
  dr = contract_DR_corner(Adr,T2d,T3r,C3)
  rdm = rdm @ dr.T
  rdm = rdm.reshape(T1l.shape[0]*Aul.shape[3]**2, Adl.shape[0]**2, dr.shape[0])
  del dr
  rdm = rdm.swapaxes(0,1).reshape(Adl.shape[0]**2, T1l.shape[0]*Aul.shape[3]**2*T2d.shape[0]*Adr.shape[2]**2)
  # ---0          ---1
  # |   2   -->   |   2
  # |11 |         |00 |
  # -----         -----
  ur = np.tensordot(C2, T1r, ((1,),(0,)))
  ur = np.tensordot(ur, T2u, ((0,),(0,)))
  ur = np.tensordot(ur, Aur, ((0, 4),(2, 3)))
  ur = np.tensordot(ur, Aur.conj(), ((0, 3, 5),(2, 3, 1)))
  #   0--------
  #       || 2|
  #       ||/ |
  #  4,7======|   -->   0----
  #       ||\ |           22|
  #       || 5|             |
  #       36  1             1
  ur = ur.transpose(0,4,7,1,3,6,2,5).reshape(rdm.shape[1],Aur.shape[0]**2)
  rdm = rdm @ ur
  rdm = rdm.reshape(Adl.shape[0], Adl.shape[0], Aur.shape[0], Aur.shape[0])
  rdm = rdm.transpose(2,0,3,1).reshape(Aur.shape[0]*Adl.shape[0], Aur.shape[0]*Adl.shape[0])
  rdm /= rdm.trace()
  return rdm
