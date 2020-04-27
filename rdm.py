import numpy as np

def rdm_1x1(env,x,y):
  #
  #   C1-0   1   3-T1-0   2     1-C2
  #   |            ||              |
  #   1    -1      12              0
  #  3      0   4    0 5          6
  #   0      \ 1    -2\ 1          0
  #   |       \|       \|          |
  #   T4-1 7 4-A--2   4-A*-2 9  2-T2
  #   | \2 8   |        |    10 3/ |
  #   3        3        3          1
  #    11       12   13             14
  #   0           01               0
  #   |           ||               |
  #   C4-1  15  3-T3-2   16     1-C3

  # assume chi is the same everywhere (not actually compulsory in CTMRG iterations)
  A = env.get_A(x+1,y+1)
  d = A.shape[0]
  D = A.shape[1]
  C1 = env.get_C1(x,y)
  chi = C1.shape[0]
  T1 = env.get_T1(x+1,y)
  T1 = T1.reshape(T1.shape[0],D,D,T1.shape[2])
  C2 = env.get_C2(x+2,y)
  T2 = env.get_T2(x+2,y+1)
  T2 = T2.reshape(T2.shape[0],T2.shape[1],D,D)
  C3 = env.get_C3(x+2,y+2)
  T3 = env.get_T3(x+1,y+2)
  T3 = T3.reshape(D,D,T3.shape[1],T3.shape[2])
  C4 = env.get_C4(x,y+2)
  T4 = env.get_T4(x,y+1)
  T4 = T4.reshape(T4.shape[0],D,D,T4.shape[2])

  T1C1 = np.dot(T1.reshape(D**2*chi, chi),C1).reshape(chi, D, D, chi)
  T2C2 = T2.transpose(1, 2, 3, 0).reshape(D**2*chi, chi)
  T2C2 = np.dot(T2C2,C2).reshape(chi, D, D, chi)
  T3C3 = T3.transpose(0, 1, 3, 2).reshape(D**2*chi, chi)
  T3C3 = np.dot(T3C3,C3.T).reshape(D, D, chi, chi)
  rdm = np.dot(T4.reshape(D**2*chi, chi),C4).reshape(chi, D, D, chi)
  T3C3 = T3C3.transpose(2, 0, 1, 3).reshape(chi, D**2*chi)
  rdm = np.dot(rdm.reshape(D**2*chi, chi),T3C3).reshape(chi, D, D, D, D, chi)
  del T3C3
  rdm = rdm.transpose(3, 1, 0, 2, 4, 5).reshape(D**2, chi**2*D**2)
  rdm = np.dot(A.reshape(d*D**2,D**2),rdm).reshape(d,D,D,chi, D, D, chi)
  T2C2T1C1 = np.dot(T2C2.reshape(D**2*chi, chi),T1C1.reshape(chi, D**2*chi)).reshape(chi, D, D, D, D, chi)
  del T2C2, T1C1
  T2C2T1C1 = T2C2T1C1.transpose(2, 4, 5, 3, 1, 0).reshape(D**2, D**2*chi**2)
  rdm = rdm.transpose(3, 1, 2, 6, 4, 5, 0).reshape(D**2*chi**2, D**2*d)
  rdm = np.dot(T2C2T1C1,rdm).reshape(D, D, D, D, d)
  del T2C2T1C1
  rdm = rdm.transpose(4, 1, 0, 3, 2).reshape(d, D**4)
  rdm = np.dot(rdm,A.transpose(1, 2, 3, 4, 0).conj().reshape(D**4, d)).reshape(d, d)
  return rdm/np.trace(rdm)


def rdm_1x2(env,x,y):
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

  Al = env.get_A(x+1,y+1)
  Ar = env.get_A(x+2,y+1)
  C1 = env.get_C1(x,y)
  d = Al.shape[0]
  D = Al.shape[3]
  chi = C1.shape[0]
  T1l = env.get_T1(x+1,y)
  D2 = T1l.shape[1]
  T1r = env.get_T1(x+2,y)
  T1r = np.dot(env.get_C2(x+3,y),T1r.reshape(chi,D**2*chi))  # include C2 here
  T2 = env.get_T2(x+3,y+1)
  C3 = env.get_C3(x+3,y+2)
  T3r = env.get_T3(x+2,y+2)
  T3l = env.get_T3(x+1,y+2)
  C4 = env.get_C4(x,y+2)
  T4 = env.get_T4(x,y+1)

  left = np.dot(C1,T4.reshape(chi,D2*chi)).reshape(chi*D2, chi)
  left = np.dot(left,C4).reshape(chi*D2, chi)
  left = np.dot(left,T3l.transpose(2, 0, 1).reshape(chi, D2*chi)).reshape(chi, D, D, D, D, chi)
  left = left.transpose(3, 1, 0, 5, 4, 2).reshape(D2,chi**2*D2)
  #   L-0            L-0
  #   L              L
  #   L=1,2   =>     L=1,2
  #   L              L
  #   L  34          L  34
  #   LLLLL-5        LLLLLL-5
  left = np.dot(Al.reshape(d*D2,D2),left).reshape(d*D2*chi**2, D2)
  left  = np.dot(left,Al.conj().transpose(3, 4, 0, 1, 2).reshape(D2, D2*d)).reshape(d,D,D,chi,chi,d,D,D)
  #     L-3               L-2
  #   0 L 16            3 L 01
  #    \LLLL=27   =>     \LLLL=47
  #    /L                /L
  #   5 L-4             6 L-5
  left = left.transpose(1,6,3,0,2,4,5,7).reshape(D2*chi,d**2*chi*D2)
  left = np.dot(T1l.reshape(chi,D2*chi),left).reshape(chi, d, D, chi, d, D)
  right = np.dot(C3.T,T2.swapaxes(0,1).reshape(chi, D2*chi))
  #   1           0
  #  2R           ||
  #  0R         2-T3r-1
  right = np.dot(T3r.swapaxes(1,2).reshape(D**2*chi,chi), right).reshape(D, D, chi, chi, D, D)
  right = right.transpose(0, 4, 2, 3, 5, 1).reshape(D2,chi**2*D2)
  #          3    0                   3    0     2
  #         4R     \1                1R     \1    \3
  #         5R     4A2      =>       4R     2A4   4A*0
  #      01  R      3              05 R      3     1
  #    2-RRRRR                   2RRRRR
  right = np.dot(Ar.swapaxes(2,4).reshape(D2*d, D2), right).reshape(d*D2*chi**2,D2)
  right = np.dot(right,Ar.conj().transpose(2, 3, 0, 1, 4).reshape(D2, d*D2)).reshape(d,D,D,chi,chi,d,D,D)
  #           4 0                 5 0
  #        16 R/               67 R/
  #   2,7=RRRRR\    =>    3,4=RRRRR\
  #           R 5                 R 1
  #      3RRRRR              2RRRRR
  right = right.transpose(0,5,3,2,7,4,1,6).reshape(D2*chi*d**2, D2*chi)
  right = np.dot(right,T1r.reshape(chi*D2,chi)).reshape(d**2, D2*chi**2)
  #     L-0           L-3        5-R
  # 1,4=L=2,5  => 3,4=L=1,2    3,4=R=0,1
  #     L-3           L-0        2-R
  left = left.transpose(3,2,5,0,1,4).reshape(D2*chi**2,d**2)
  rdm = np.dot(right,left).reshape(d, d, d, d)
  del right,left
  rdm = rdm.transpose(2,0,1,3).reshape(d**2,d**2)
  rdm = rdm/np.trace(rdm)
  return rdm


def rdm_2x1(env,x,y):
  # CPU optimal, not memory optimal
  # assume chi is the same everywhere (not actually compulsory in CTMRG iterations)
  #
  #   C1-0     2-T1-0       1-C2
  #   |          |             |
  #   1          1             0
  #        0       0
  #   0     \ 1     \ 1        0
  #   |      \|      \|        |
  #   T4-1  4-A--2  4-A*-2  2-T2
  #   |       |       |        |
  #   2       3       3        1
  #        0       0
  #   0     \ 1     \ 1        0
  #   |      \|      \|        |
  #   T4-1  4-A--2  4-A*-2  2-T2
  #   |       |       |        |
  #   2       3       3        1
  #
  #   0          0             0
  #   |          |             |
  #   C4-1     2-T3-1       1-C3

  # assume chi is the same everywhere (not actually compulsory in CTMRG iterations)
  Au = env.get_A(x+1,y+1)
  Ad = env.get_A(x+1,y+2)
  C1 = env.get_C1(x,y)
  d = Au.shape[0]
  D = Au.shape[1]
  chi = C1.shape[0]
  T1 = env.get_T1(x+1,y)
  T1 = T1.reshape(T1.shape[0],D,D,T1.shape[2])
  C2 = env.get_C2(x+2,y)
  T2u = env.get_T2(x+2,y+1)
  T2u = T2u.reshape(T2u.shape[0],T2u.shape[1],D,D)
  T2d = env.get_T2(x+2,y+2)
  T2d = T2d.reshape(T2d.shape[0],T2d.shape[1],D,D)
  C3 = env.get_C3(x+2,y+3)
  T3 = env.get_T3(x+1,y+3)
  T3 = T3.reshape(D,D,T3.shape[1],T3.shape[2])
  C4 = env.get_C4(x,y+3)
  T4d = env.get_T4(x,y+2)
  T4d = T4d.reshape(T4d.shape[0],D,D,T4d.shape[2])
  T4u = env.get_T4(x,y+1)
  T4u = T4u.reshape(T4u.shape[0],D,D,T4u.shape[2])

  up = np.dot(C2,T1.reshape(chi, D**2*chi)).reshape(chi, D, D, chi)
  down = T3.transpose(0, 1, 3, 2).reshape(D**2*chi, chi)
  down = np.dot(down,C3.T).reshape(D, D, chi, chi)
  down = down.transpose(0, 1, 3, 2).reshape(D**2*chi, chi)
  down = np.dot(down,C4.T).reshape(D, D, chi, chi)
  temp = T4d.transpose(3, 0, 1, 2).reshape(chi, D**2*chi)
  down = np.dot(down.reshape(D**2*chi, chi),temp).reshape(D, D, chi, chi, D, D)
  down = down.transpose(1, 2, 3, 5, 0, 4).reshape(D**2*chi**2, D**2)
  temp = Ad.transpose(3, 4, 0, 1, 2).reshape(D**2, D**2*d)
  down = np.dot(down,temp).reshape(D, chi, chi, D, d, D, D)
  down = down.transpose(1, 2, 4, 5, 6, 0, 3).reshape(D**2*chi**2*d, D**2)
  temp = Ad.conj().transpose(3, 4, 0, 1, 2).reshape(D**2, D**2*d)
  down = np.dot(down,temp).reshape(chi, chi, d, D, D, d, D, D)
  down = down.transpose(1, 2, 3, 5, 6, 0, 4, 7).reshape(D**2*chi*d**2, D**2*chi)
  temp = T2d.transpose(1, 2, 3, 0).reshape(D**2*chi, chi)
  down = np.dot(down,temp).reshape(chi, d, D, d, D, chi)
  up = up.transpose(1, 2, 3, 0).reshape(D**2*chi, chi)
  up = np.dot(up,T2u.reshape(chi, D**2*chi)).reshape(D, D, chi, chi, D, D)
  up = up.transpose(0, 2, 3, 4, 1, 5).reshape(D**2*chi**2, D**2)
  temp = Au.conj().transpose(1, 2, 0, 3, 4).reshape(D**2, D**2*d)
  up = np.dot(up,temp).reshape(D, chi, chi, D, d, D, D)
  up = up.transpose(1, 2, 4, 5, 6, 0, 3).reshape(D**2*chi**2*d, D**2)
  temp = Au.transpose(1, 2, 0, 3, 4).reshape(D**2, D**2*d)
  up = np.dot(up,temp).reshape(chi, chi, d, D, D, d, D, D)
  up = up.transpose(1, 2, 3, 5, 6, 0, 7, 4).reshape(D**2*chi*d**2, D**2*chi)
  up = np.dot(up,T4u.reshape(D**2*chi, chi)).reshape(chi, d, D, d, D, chi)
  up = up.transpose(1, 3, 5, 4, 2, 0).reshape(d**2, D**2*chi**2)
  down = down.transpose(0, 2, 4, 5, 1, 3).reshape(D**2*chi**2, d**2)
  rdm = np.dot(up,down).reshape(d, d, d, d)
  del up,down

  rdm = rdm.transpose(1, 2, 0, 3).reshape(d**2,d**2)
  rdm = rdm/np.trace(rdm)
  return rdm


def rdm_2x2(env,x,y):
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


  Aul = env.get_A(x+1,y+1)
  Aur = env.get_A(x+2,y+1)
  Adr = env.get_A(x+2,y+2)
  Adl = env.get_A(x+1,y+2)
  C1 = env.get_C1(x,y)
  d = Aul.shape[0]
  D = Aul.shape[1]
  chi = C1.shape[0]
  T1l = env.get_T1(x+1,y)
  T1l = T1l.reshape(T1l.shape[0],D,D,T1l.shape[2])
  T1r = env.get_T1(x+2,y)
  T1r = T1r.reshape(T1r.shape[0],D,D,T1r.shape[2])
  C2 = env.get_C2(x+2,y)
  T2u = env.get_T2(x+3,y+1)
  T2u = T2u.reshape(T2u.shape[0],T2u.shape[1],D,D)
  T2d = env.get_T2(x+3,y+2)
  T2d = T2d.reshape(T2d.shape[0],T2d.shape[1],D,D)
  C3 = env.get_C3(x+3,y+3)
  T3r = env.get_T3(x+2,y+3)
  T3r = T3r.reshape(D,D,T3r.shape[1],T3r.shape[2])
  T3l = env.get_T3(x+1,y+3)
  T3l = T3l.reshape(D,D,T3l.shape[1],T3l.shape[2])
  C4 = env.get_C4(x,y+3)
  T4d = env.get_T4(x,y+2)
  T4d = T4d.reshape(T4d.shape[0],D,D,T4d.shape[2])
  T4u = env.get_T4(x,y+1)
  T4u = T4u.reshape(T4u.shape[0],D,D,T4u.shape[2])

  ul = np.dot(T1l.reshape(D**2*chi, chi),C1).reshape(chi, D, D, chi)
  ur = np.dot(C2,T1r.reshape(chi, D**2*chi)).reshape(chi, D, D, chi)
  dr = np.dot(C3,T3r.transpose(2, 0, 1, 3).reshape(chi, D**2*chi)).reshape(chi, D, D, chi)
  dl = np.dot(T4d.reshape(D**2*chi, chi),C4).reshape(chi, D, D, chi)
  dl = np.dot(dl.reshape(D**2*chi, chi),T3l.transpose(3, 0, 1, 2).reshape(chi, D**2*chi)).reshape(chi, D, D, D, D, chi)
  dl = dl.transpose(0, 2, 4, 5, 1, 3).reshape(D**2*chi**2, D**2)
  dl = np.dot(dl,Adl.transpose(4, 3, 0, 1, 2).reshape(D**2, D**2*d)).reshape(chi, D, D, chi, d, D, D)
  dl = dl.transpose(0, 3, 4, 5, 6, 1, 2).reshape(D**2*chi**2*d, D**2)
  dl = np.dot(dl,Adl.conj().transpose(4, 3, 0, 1, 2).reshape(D**2, D**2*d)).reshape(chi, chi, d, D, D, d, D, D)
  ul = np.dot(ul.reshape(D**2*chi, chi),T4u.reshape(chi, D**2*chi)).reshape(chi, D, D, D, D, chi)
  ul = ul.transpose(0, 1, 3, 5, 2, 4).reshape(D**2*chi**2, D**2)
  ul = np.dot(ul,Aul.conj().transpose(1, 4, 0, 2, 3).reshape(D**2, D**2*d)).reshape(chi, D, D, chi, d, D, D)
  ul = ul.transpose(0, 3, 4, 5, 6, 1, 2).reshape(D**2*chi**2*d, D**2)
  ul = np.dot(ul,Aul.transpose(1, 4, 0, 2, 3).reshape(D**2, D**2*d)).reshape(chi, chi, d, D, D, d, D, D)
  ul = ul.transpose(0, 2, 3, 5, 6, 1, 7, 4).reshape(D**2*chi*d**2, D**2*chi)
  dl = dl.transpose(0, 3, 6, 1, 2, 4, 5, 7).reshape(D**2*chi, D**2*chi*d**2)
  left = np.dot(ul,dl).reshape(chi, d, D, d, D, chi, d, D, d, D)
  del ul,dl
  ur = ur.transpose(1, 2, 3, 0).reshape(D**2*chi, chi)
  ur = np.dot(ur,T2u.reshape(chi, D**2*chi)).reshape(D, D, chi, chi, D, D)
  dr = dr.transpose(1, 2, 3, 0).reshape(D**2*chi, chi)
  dr = np.dot(dr,T2d.transpose(1, 0, 2, 3).reshape(chi, D**2*chi)).reshape(D, D, chi, chi, D, D)
  ur = ur.transpose(1, 2, 3, 5, 0, 4).reshape(D**2*chi**2, D**2)
  ur = np.dot(ur,Aur.transpose(1, 2, 0, 3, 4).reshape(D**2, D**2*d)).reshape(D, chi, chi, D, d, D, D)
  dr = dr.transpose(1, 2, 3, 5, 4, 0).reshape(D**2*chi**2, D**2)
  dr = np.dot(dr,Adr.transpose(2, 3, 0, 1, 4).reshape(D**2, D**2*d)).reshape(D, chi, chi, D, d, D, D)
  ur = ur.transpose(1, 2, 4, 5, 6, 0, 3).reshape(D**2*chi**2*d, D**2)
  ur = np.dot(ur,Aur.conj().transpose(1, 2, 0, 3, 4).reshape(D**2, D**2*d)).reshape(chi, chi, d, D, D, d, D, D)
  dr = dr.transpose(1, 2, 4, 5, 6, 3, 0).reshape(D**2*chi**2*d, D**2)
  dr = np.dot(dr,Adr.conj().transpose(2, 3, 0, 1, 4).reshape(D**2, D**2*d)).reshape(chi, chi, d, D, D, d, D, D)
  dr = dr.transpose(0, 2, 4, 5, 7, 3, 6, 1).reshape(D**2*chi*d**2, D**2*chi)
  ur = ur.transpose(3, 6, 1, 0, 2, 4, 5, 7).reshape(D**2*chi, D**2*chi*d**2)
  right = np.dot(dr,ur).reshape(chi, d, D, d, D, chi, d, D, d, D)
  del dr, ur
  right = right.transpose(1, 3, 6, 8, 5, 7, 9, 2, 4, 0).reshape(d**4, D**4*chi**2)
  left = left.transpose(0, 4, 2, 7, 9, 5, 1, 3, 6, 8).reshape(D**4*chi**2, d**4)
  rdm = np.dot(right,left).reshape(d, d, d, d, d, d, d, d)


  rdm = rdm.transpose(5, 2, 6, 0, 4, 3, 7, 1).reshape(d**4,d**4)
  rdm = rdm/np.trace(rdm)
  return rdm
