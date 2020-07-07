import numpy as np

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
  T3C3 = T3.swapaxes(1,2).reshape(T3.shape[0]*T3.shape[2], T3.shape[1])
  T3C3 = np.dot(T3C3,C3.T).reshape(T3.shape[0], T3.shape[2], C3.shape[0])
  T3C3 = T3C3.swapaxes(0,1).reshape(T3.shape[2], T3.shape[0]*C3.shape[0])
  rdm = np.dot(T4.reshape(T4.shape[0]*T4.shape[1], T4.shape[2]),C4)
  rdm = np.dot(rdm,T3C3).reshape(T4.shape[0], A.shape[4], A.shape[4], A.shape[3], A.shape[3]*C3.shape[0])
  del T3C3
  rdm = rdm.transpose(3, 1, 0, 2, 4).reshape(A.shape[3]*A.shape[4], T4.shape[0]*A.shape[4]*A.shape[3]*C3.shape[0])
  rdm = np.dot(A.reshape(A.shape[0]*A.shape[1]*A.shape[2]*A.shape[3], rdm.shape[1]),rdm)
  rdm = rdm.reshape(A.shape[0], A.shape[1]*A.shape[2]*A.shape[3], T4.shape[0], A.shape[4]*A.shape[3], C3.shape[0])
  rdm = rdm.transpose(2, 1, 4, 3, 0).reshape(T4.shape[0]*A.shape[1]*A.shape[2]*C3.shape[0], A.shape[4]*A.shape[3]*A.shape[0])
  T1C1 = np.dot(T1.reshape(T1.shape[0]*T1.shape[1], T1.shape[2]),C1).reshape(T1.shape[0], T1.shape[1]*C1.shape[1])
  T2C2 = T2.transpose(1, 2, 0).reshape(T2.shape[1]*T2.shape[2], T2.shape[0])
  T2C2 = np.dot(T2C2,C2)
  T2C2T1C1 = np.dot(T2C2,T1C1).reshape(T2.shape[1], A.shape[2], A.shape[2], A.shape[1], A.shape[1]*C1.shape[1])
  del T2C2, T1C1
  T2C2T1C1 = T2C2T1C1.transpose(2, 4, 3, 1, 0).reshape(A.shape[2]*A.shape[1], rdm.shape[0])
  rdm = np.dot(T2C2T1C1,rdm).reshape(A.shape[2], A.shape[1], A.shape[4], A.shape[3], A.shape[0])
  del T2C2T1C1
  rdm = rdm.transpose(4, 1, 0, 3, 2).reshape(A.shape[0], A.shape[1]*A.shape[2]*A.shape[3]*A.shape[4])
  rdm = np.dot(rdm,A.reshape(A.shape[0],rdm.shape[1]).T.conj()).reshape(A.shape[0], A.shape[0])
  return rdm/np.trace(rdm)


def rdm_1x2(C1, T1l, T1r, C2, T4, Al, Ar, T2, C4, T3l, T3r, C3):
  """
  Compute reduced density matrix for 2 sites in a row
  """
  # CPU optimal, not memory optimal
  # assume chi is the same everywhere (not actually compulsory in CTMRG iterations)

  #
  #   C1-0     2-T1-0           2-T1-0       1-C2
  #   |          |                |             |
  #   1          1                1             0
  #        0       0        0       0
  #   0     \ 1     \ 1      \ 1     \ 1        0
  #   |      \|      \|       \|      \|        |
  #   T4-1  4-A--2  4-A*-2   4-A--2  4-A*-2  2-T2
  #   |       |       |        |       |        |
  #   2       3       3        3       3        1
  #
  #   0          0                0             0
  #   |          |                |             |
  #   C4-1     2-T3-1           2-T3-1       1-C3

  left = np.dot(C1,T4.reshape(T4.shape[0],T4.shape[1]*T4.shape[2])).reshape(C1.shape[0]*T4.shape[1], T4.shape[2])
  left = np.dot(left,C4)
  left = np.dot(left,T3l.transpose(2, 0, 1).reshape(T3l.shape[2], T3l.shape[0]*T3l.shape[1])).reshape(C1.shape[0], Al.shape[4], Al.shape[4], Al.shape[3], Al.shape[3], T3l.shape[1])
  left = left.transpose(3, 1, 0, 5, 4, 2).reshape(Al.shape[3]*Al.shape[4], C1.shape[0]*T3l.shape[1]*Al.shape[3]*Al.shape[4])
  #   L-0            L-2
  #   L              L
  #   L=1,2   =>     L=1,5
  #   L              L
  #   L  34          L  04
  #   LLLLL-5        LLLLLL-3
  left = np.dot(Al.reshape(Al.shape[0]*Al.shape[1]*Al.shape[2], Al.shape[3]*Al.shape[4]),left).reshape(Al.shape[0]*Al.shape[1]*Al.shape[2]*C1.shape[0]*T3l.shape[1], Al.shape[3]*Al.shape[4])
  left  = np.dot(left,Al.reshape(Al.shape[0]*Al.shape[1]*Al.shape[2], Al.shape[3]*Al.shape[4]).conj().T)
  left = left.reshape(Al.shape[0], Al.shape[1], Al.shape[2], C1.shape[0], T3l.shape[1]*Al.shape[0], Al.shape[1], Al.shape[2])
  #     L-3               L-2
  #   0 L 16            3 L 01
  #    \LLLL=27   =>     \LLLL=47
  #    /L                /L
  #   5 L-4             6 L-5
  left = left.transpose(1,5,3,0,2,4,6).reshape(T1l.shape[1]*T1l.shape[2], (Al.shape[0]*Al.shape[2])**2*T3l.shape[1])
  left = np.dot(T1l.reshape(T1l.shape[0],T1l.shape[1]*T1l.shape[2]),left).reshape(T1l.shape[0], Al.shape[0], Al.shape[2], T3l.shape[1], Al.shape[0], Al.shape[2])
  #     L-0           L-3        5-R
  # 1,4=L=2,5  => 3,4=L=1,2    3,4=R=0,1
  #     L-3           L-0        2-R
  left = left.transpose(3,2,5,0,1,4).reshape(T3l.shape[1]*Al.shape[2]*Al.shape[2]*T1l.shape[0], Al.shape[0]**2)

  right = np.dot(C3.T,T2.swapaxes(0,1).reshape(T2.shape[1], T2.shape[0]*T2.shape[2]))
  #   1           0
  #  2R           ||
  #  0R         2-T3r-1
  right = np.dot(T3r.swapaxes(1,2).reshape(T3r.shape[0]*T3r.shape[2],T3r.shape[1]), right)
  right = right.reshape(Ar.shape[3], Ar.shape[3], T3r.shape[2]*T2.shape[0], Ar.shape[2], Ar.shape[2])
  right = right.transpose(0, 3, 2, 4, 1).reshape(Ar.shape[3]*Ar.shape[2], T3r.shape[2]*T2.shape[0]*Ar.shape[2]*Ar.shape[3])
  #          3    0                   3    0     2
  #         4R     \1                1R     \1    \3
  #         5R     4A2      =>       4R     2A4   4A*0
  #      01  R      3              05 R      3     1
  #    2-RRRRR                   2RRRRR
  right = np.dot(Ar.swapaxes(2,4).reshape(Ar.shape[0]*Ar.shape[1]*Ar.shape[4], right.shape[0]), right).reshape(Ar.shape[0]*Ar.shape[1]*Ar.shape[4]*T3r.shape[2]*T2.shape[0], Ar.shape[2]*Ar.shape[3])
  right = np.dot(right,Ar.conj().transpose(2, 3, 0, 1, 4).reshape(right.shape[1], Ar.shape[0]*Ar.shape[1]*Ar.shape[4]))
  right = right.reshape(Ar.shape[0], Ar.shape[1], Ar.shape[4], T3r.shape[2], T2.shape[0], Ar.shape[0], Ar.shape[1], Ar.shape[4])
  #           4 0                 5 0
  #        16 R/               67 R/
  #   2,7=RRRRR\    =>    3,4=RRRRR\
  #           R 5                 R 1
  #      3RRRRR              2RRRRR
  right = right.transpose(0,5,3,2,7,4,1,6).reshape((Ar.shape[0]*Ar.shape[4])**2*T3r.shape[2], T2.shape[0]*Ar.shape[1]**2)
  T1rC1 = np.dot(C2,T1r.reshape(C2.shape[1],T1r.shape[1]*T1r.shape[2])).reshape(C2.shape[0]*T1r.shape[1], T1r.shape[2])
  right = np.dot(right,T1rC1).reshape(Ar.shape[0]**2, Ar.shape[4]**2*T3r.shape[2]*T1r.shape[2])
  rdm = np.dot(right,left).reshape(Ar.shape[0],Ar.shape[0],Al.shape[0],Al.shape[0])
  del right,left
  rdm = rdm.transpose(2,0,3,1).reshape(Al.shape[0]*Ar.shape[0],Al.shape[0]*Ar.shape[0])
  rdm = rdm/np.trace(rdm)
  return rdm


def rdm_2x1(C1,T1,C2,T4u,Au,T2u,T4d,Ad,T2d,C4,T3,C3):
  """
  Compute reduced density matrix for 2 sites in a column
  """
  # contract using 1x2 with swapped tensors and legs
  return rdm_1x2(C2, T2u.transpose(1,2,0), T2d.transpose(1,2,0), C3.T, T1,
                 Au.transpose(0,2,3,4,1), Ad.transpose(0,2,3,4,1),
                 T3.transpose(1,2,0), C1, T4u.transpose(1,2,0),
                 T4d.transpose(1,2,0),C4.T)


def rdm_2x2(C1,T1l,T1r,C2,T4u,Aul,Aur,T2u,T4d,Adl,Adr,T2d,C4,T3l,T3r,C3):
  #
  #   C1-0     2-T1-0           2-T1-0       1-C2
  #   |          |                |             |
  #   1          1                1             0
  #        0       0        0       0
  #   0     \ 1     \ 1      \ 1     \ 1        0
  #   |      \|      \|       \|      \|        |
  #   T4-1  4-A--2  4-A*-2   4-A--2  4-A*-2  2-T2
  #   |       |       |        |       |        |
  #   2       3       3        3       3        1
  #
  #        0       0        0       0
  #   0     \ 1     \ 1      \ 1     \ 1        0
  #   |      \|      \|       \|      \|        |
  #   T4-1  4-A--2  4-A*-2   4-A--2  4-A*-2  2-T2
  #   |       |       |        |       |        |
  #   2       3       3        3       3        1
  #   0          0                0             0
  #   |          |                |             |
  #   C4-1     2-T3-1           2-T3-1       1-C3

  Adl2 = Adl.transpose(4, 3, 0, 1, 2).reshape(Adl.shape[4]*Adl.shape[3], Adl.shape[0]*Adl.shape[1]*Adl.shape[2])
  Aul2 = Aul.transpose(1, 4, 0, 2, 3).reshape(Aul.shape[1]*Aul.shape[4], Aul.shape[0]*Aul.shape[2]*Aul.shape[3])
  Aur2 = Aur.transpose(1, 2, 0, 3, 4).reshape(Aur.shape[1]*Aur.shape[2], Aur.shape[0]*Aur.shape[3]*Aur.shape[4])
  Adr2 = Adr.transpose(2, 3, 0, 1, 4).reshape(Adr.shape[2]*Adr.shape[3], Adr.shape[0]*Adr.shape[1]*Adr.shape[4])

  dl = np.dot(T4d.reshape(T4d.shape[0]*T4d.shape[1], T4d.shape[2]),C4)
  dl = np.dot(dl,T3l.transpose(2, 0, 1).reshape(T3l.shape[2], T3l.shape[0]*T3l.shape[1])).reshape(T4d.shape[0], Adl.shape[4], Adl.shape[4], Adl.shape[3], Adl.shape[3]*T3l.shape[1])
  dl = dl.transpose(0, 2, 4, 1, 3).reshape(T4d.shape[0]*Adl.shape[4]*Adl.shape[3]*T3l.shape[1], Adl2.shape[0])
  dl = np.dot(dl,Adl2).reshape(T4d.shape[0], Adl.shape[4]*Adl.shape[3], T3l.shape[1]*Adl2.shape[1])
  dl = dl.swapaxes(1,2).reshape(T4d.shape[0]*T3l.shape[1]*Adl2.shape[1], Adl2.shape[0])
  dl = np.dot(dl,Adl2.conj()).reshape(T4d.shape[0], T3l.shape[1]*Adl.shape[0], Adl.shape[1], Adl.shape[2]*Adl.shape[0], Adl.shape[1],Adl.shape[2])
  dl = dl.transpose(0, 2, 4, 1, 3, 5).reshape(T4d.shape[0]*Adl.shape[1]**2, T3l.shape[1]*(Adl.shape[0]*Adl.shape[2])**2)

  ul = np.dot(T1l.reshape(T1l.shape[0]*T1l.shape[1], T1l.shape[2]),C1)
  ul = np.dot(ul,T4u.reshape(T4u.shape[0], T4u.shape[1]*T4u.shape[2])).reshape(T1l.shape[0]*Aul.shape[1], Aul.shape[1], Aul.shape[4], Aul.shape[4], T4u.shape[2])
  ul = ul.transpose(0, 2, 4, 1, 3).reshape(T1l.shape[0]*Aul2.shape[0]*T4u.shape[2], Aul2.shape[0])
  ul = np.dot(ul,Aul2.conj()).reshape(T1l.shape[0], Aul2.shape[0], T4u.shape[2]*Aul2.shape[1])
  ul = ul.swapaxes(1,2).reshape(T1l.shape[0]*T4u.shape[2]*Aul2.shape[1], Aul2.shape[0])
  ul = np.dot(ul,Aul2).reshape(T1l.shape[0], T4u.shape[2], Aul.shape[0]*Aul.shape[2], Aul.shape[3], Aul.shape[0]*Aul.shape[2], Aul.shape[3])
  ul = ul.transpose(0, 2, 4, 1, 5, 3).reshape(T1l.shape[0]*(Aul.shape[0]*Aul.shape[2])**2, T4u.shape[2]*Aul.shape[3]**2)
  left = np.dot(ul,dl).reshape(T1l.shape[0], Aul.shape[0], Aul.shape[2], Aul.shape[0], Aul.shape[2], T3l.shape[1], Adl.shape[0], Adl.shape[2], Adl.shape[0], Adl.shape[2])
  del ul,dl
  left = left.transpose(0, 4, 2, 7, 9, 5, 1, 3, 6, 8).reshape(T1l.shape[0]*Aul.shape[2]**2*Adl.shape[2]**2*T3l.shape[1], (Aul.shape[0]*Adl.shape[0])**2)

  ur = np.dot(C2,T1r.reshape(T1r.shape[0], T1r.shape[1]*T1r.shape[2]))
  ur = np.dot(ur.T,T2u.reshape(T2u.shape[0], T2u.shape[1]*T2u.shape[2])).reshape(Aur.shape[1], Aur.shape[1]*T1r.shape[2]*T2u.shape[1], Aur.shape[2], Aur.shape[2])
  ur = ur.transpose(1, 3, 0, 2).reshape(Aur.shape[1]*T1r.shape[2]*T2u.shape[1]*Aur.shape[2], Aur2.shape[0])
  ur = np.dot(ur,Aur2).reshape(Aur.shape[1], T1r.shape[2]*T2u.shape[1], Aur.shape[2], Aur2.shape[1])
  ur = ur.transpose(1, 3, 0, 2).reshape(T1r.shape[2]*T2u.shape[1]*Aur2.shape[1], Aur2.shape[0])
  ur = np.dot(ur,Aur2.conj()).reshape(T1r.shape[2], T2u.shape[1], Aur.shape[0], Aur.shape[3], Aur.shape[4]*Aur.shape[0], Aur.shape[3], Aur.shape[4])
  ur = ur.transpose(3, 5, 1, 0, 2, 4, 6).reshape(Aur.shape[3]**2*T2u.shape[1], T1r.shape[2]*(Aur.shape[0]*Aur.shape[4])**2)
  dr = np.dot(C3,T3r.swapaxes(0,1).reshape(T3r.shape[1], T3r.shape[0]*T3r.shape[2]))
  dr = np.dot(dr.T,T2d.swapaxes(0,1).reshape(T2d.shape[1], T2d.shape[0]*T2d.shape[2])).reshape(Adr.shape[3], Adr.shape[3]*T3r.shape[2]*T2d.shape[0], Adr.shape[2], Adr.shape[2])
  dr = dr.transpose(1, 3, 2, 0).reshape(Adr.shape[3]*T3r.shape[2]*T2d.shape[0]*Adr.shape[2], Adr2.shape[0])
  dr = np.dot(dr,Adr2).reshape(Adr.shape[3], T3r.shape[2]*T2d.shape[0], Adr.shape[2], Adr2.shape[1])
  dr = dr.transpose(1, 3, 2, 0).reshape(T3r.shape[2]*T2d.shape[0]*Adr2.shape[1], Adr2.shape[0])
  dr = np.dot(dr,Adr2.conj()).reshape(T3r.shape[2], T2d.shape[0], Adr.shape[0], Adr.shape[1], Adr.shape[4]*Adr.shape[0], Adr.shape[1], Adr.shape[4])
  dr = dr.transpose(0, 2, 4, 6, 3, 5, 1).reshape(T3r.shape[2]*(Adr.shape[4]*Adr.shape[0])**2, ur.shape[0])
  right = np.dot(dr,ur).reshape(T3r.shape[2], Adr.shape[0], Adr.shape[4], Adr.shape[0], Adr.shape[4], T1r.shape[2], Aur.shape[0], Aur.shape[4], Aur.shape[0], Aur.shape[4])
  del dr, ur
  right = right.transpose(1, 3, 6, 8, 5, 7, 9, 2, 4, 0).reshape((Adr.shape[0]*Aur.shape[0])**2, left.shape[0])
  rdm = np.dot(right,left).reshape(Adr.shape[0], Adr.shape[0], Aur.shape[0], Aur.shape[0], Aul.shape[0], Aul.shape[0], Adl.shape[0], Adl.shape[0])
  del right, left

  rdm = rdm.transpose(5, 2, 6, 0, 4, 3, 7, 1).reshape(Aul.shape[0]*Aur.shape[0]*Adl.shape[0]*Adr.shape[0], Aul.shape[0]*Aur.shape[0]*Adl.shape[0]*Adr.shape[0])
  rdm = rdm/np.trace(rdm)
  return rdm
