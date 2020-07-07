import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB
from toolsU1 import checkU1, svdU1
from test_tools import construct_genSU2_s


D = 7
d = 2
A = np.zeros((d,d,D,D,D,D))
A[0,0,0,0,0,0] = 1
A[1,1,0,0,0,0] = 1
B = A.copy()

p_col = np.array([1,-1], dtype=np.int8)
v_col = np.zeros(D,dtype=np.int8)

gen2 = construct_genSU2_s(1/2)
h = np.tensordot(gen2, -gen2.conj(), (0,0)).real.swapaxes(1,2).reshape(4,4)
tau = 0.1

su = SimpleUpdateAB(A.shape, h, tau, A0=A, B0=B)
su._colorsA = [-p_col,p_col] + [v_col]*4
su._colorsB = [p_col,-p_col] + [v_col]*4   # rotation on physical and ancilla legs
print(1, checkU1(su._gammaA,su._colorsA))
print(2, checkU1(su._gammaB,su._colorsB))

def move_right(su):
  M_A0 = np.einsum('paurdl,u,d,l->audlpr', su._gammaA, su._lambda_u, su._lambda_d, su._lambda_l)
  M_B0 = np.einsum('paurdl,u,r,d->lpaurd', su._gammaB, su._lambda_d, su._lambda_l, su._lambda_u)
  lambda_dir = su._lambda_r
  D_cst = D**3*d
  D_dir = D

  M_A0 = M_A0.reshape(D_cst, D_dir*d)
  colorsMA = [(su._colorsA[1][:,None,None,None] + su._colorsA[2][:,None,None] + su._colorsA[4][:,None] + su._colorsA[5]).ravel(),
              (su._colorsA[0][:,None] + su._colorsA[3]).ravel()]
  print(3, checkU1(M_A0,colorsMA))
  W_A, sA, M_A, colors_sA = svdU1(M_A0, colorsMA[0], -colorsMA[1])
  D_effA = len(sA)  # not D_dir*d if colors do not match
  M_A *= sA[:,None]
  print(f"svd A: {lg.norm(W_A @ M_A - M_A0)/lg.norm(M_A0):.1e}")

  M_B0 = M_B0.reshape(D_dir*d, D_cst)
  colorsMB = [(su._colorsB[5][:,None] + su._colorsB[0]).ravel(), (su._colorsB[1][:,None,None,None]
               + su._colorsB[2][:,None,None] + su._colorsB[3][:,None] + su._colorsB[4]).ravel()]
  print(4, checkU1(M_B0,colorsMB))
  M_B, sB, W_B, colors_sB = svdU1(M_B0, colorsMB[0], -colorsMB[1])
  D_effB = len(sB)
  M_B *= sB
  print(f"svd B: {lg.norm(M_B @ W_B - M_B0)/lg.norm(M_B0):.1e}")


  theta = M_A.reshape(D_effA*d, D_dir)
  theta *= su._lambda_r
  theta = np.dot(theta, M_B.reshape(D_dir, d*D_effB) )
  theta = theta.reshape(D_effA, d, d, D_effB).transpose(0,3,1,2).reshape(D_effA*D_effB, d**2)
  theta = np.dot(theta, su._gr)

  theta = theta.reshape(D_effA, D_effB, d, d).swapaxes(1,2).reshape(D_effA*d, D_effB*d)
  row_col = (colors_sA[:,None] + su._colorsA[0]).ravel()
  col_col = (-colors_sB[:,None] + su._colorsB[0]).ravel()
  print(5, checkU1(theta,[row_col,col_col]))
  M_A2,s,M_B2,colors_th = svdU1(theta,row_col,-col_col)
  print(f"svd theta: {lg.norm(M_A2*s @ M_B2 - theta)/lg.norm(theta):.1e}")
  print(6, checkU1(M_A2,[row_col,-colors_th]))
  print(7, checkU1(M_B2,[colors_th,col_col]))


  # 4) renormalize link dimension
  s = s[:D_dir]
  s /= s.sum()  # singular values are positive
  su._lambda_r = s
  colors_th = colors_th[:D_dir]

  # 5) start reconstruction of new gammaA and gammaB by unifying cst and eff
  M_A2 = M_A2[:,:D_dir].reshape(D_effA, d*D_dir)
  print(8, checkU1(M_A2,[colors_sA, (-p_col[:,None] - colors_th).ravel()]))
  M_A = np.dot(W_A, M_A2)
  print(9, checkU1(M_A,[colorsMA[0], (-p_col[:,None] - colors_th).ravel()]))
  M_B2 = M_B2[:D_dir].reshape(D_dir,D_effB,d).swapaxes(1,2).reshape(D_dir*d,D_effB)
  print(10, checkU1(M_B2,[(colors_th[:,None] + p_col).ravel(), -colors_sB]))
  M_B = np.dot(M_B2, W_B)
  print(11, checkU1(M_B,[(colors_th[:,None] + p_col).ravel(), colorsMB[1]]))

  M_A = M_A.reshape(su._a, su._Du, su._Dd, su._Dl, su._d, su._Dr)
  su._gammaA = np.einsum('audlpr,u,d,l->paurdl', M_A, su._lambda_u**-1, su._lambda_d**-1, su._lambda_l**-1)
  su._colorsA = [-p_col,p_col,v_col,-colors_th,v_col,v_col]
  print(12, checkU1(su._gammaA, su._colorsA))
  M_B = M_B.reshape(su._Dr, su._d, su._a, su._Dd, su._Dl, su._Du)
  su._gammaB = np.einsum('lpaurd,u,r,d->paurdl', M_B, su._lambda_d**-1, su._lambda_l**-1, su._lambda_u**-1)
  su._colorsB = [p_col,-p_col,v_col,v_col,v_col,colors_th]
  print(13, checkU1(su._gammaB, su._colorsB))


move_right(su)
move_right(su)
