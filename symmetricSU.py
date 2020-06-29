import numpy as np
import scipy.linalg as lg
from simple_updateAB import SimpleUpdateAB
from test_tools import construct_genSU2_s

def svdU1(M, row_colors, col_colors):
  """
  Singular value decomposition for a U(1) symmetric matrix M.

  Parameters
  ----------
  M : (m,n) ndarray
    Matrix to decompose.
  row_colors : (m,) integer ndarray
    U(1) quantum numbers of the rows.
  col_colors : (n,) integer ndarray
    U(1) quantum numbers of the columns.

  Returns
  -------
  U : (m,k) ndarray
    Left singular vectors.
  s : (k,) ndarray
    Singular values.
  V : (k,n) right singular vectors
  colors : (k,) integer ndarray
    U(1) quantum numbers of U columns and V rows.

  Note that k may be < min(m,n) if row and column colors do not match on more
  than min(m,n) values. If k = 0 (no matching color), an error is raised to
  avoid messy zero-length arrays (implies M=0, all singular values are 0)
  """
  if M.ndim != 2:
    raise ValueError("M has to be a matrix")
  if row_colors.shape != (M.shape[0],):
    raise ValueError("row_colors has to be (M.shape[0])")
  if col_colors.shape != (M.shape[1],):
    raise ValueError("col_colors has to be (M.shape[1])")

  row_sort = np.argsort(row_colors)
  sorted_row_colors = row_colors[row_sort]
  col_sort = np.argsort(col_colors)
  sorted_col_colors = col_colors[col_sort]
  row_inds = [0, *((sorted_row_colors[:-1] != sorted_row_colors[1:]
                    ).nonzero()[0] + 1), M.shape[0]]
  col_inds = [0, *((sorted_col_colors[:-1] != sorted_col_colors[1:]
                    ).nonzero()[0] + 1), M.shape[1]]
  dmin = min(M.shape)
  U = np.zeros((M.shape[0],dmin))
  s = np.empty(dmin)
  V = np.zeros((dmin,M.shape[1]))
  colors = np.empty(dmin,dtype=np.int8)

  # match blocks with same color and compute SVD inside those blocks only
  k,br,bc,brmax,bcmax = 0,0,0,len(row_inds)-1,len(col_inds)-1
  while br < brmax and bc < bcmax:
    if sorted_row_colors[row_inds[br]] == sorted_col_colors[col_inds[bc]]:
      ir,jr = row_inds[br:br+2]
      ic,jc = col_inds[bc:bc+2]
      m = M[row_sort[ir:jr,None], col_sort[ic:jc]]
      d = min(m.shape)
      U[row_sort[ir:jr],k:k+d], s[k:k+d], V[k:k+d,col_sort[ic:jc]] = lg.svd(
                                                         m,full_matrices=False)
      colors[k:k+d] = sorted_row_colors[row_inds[br]]
      k += d
      br += 1
      bc += 1
    elif sorted_row_colors[br] < sorted_col_colors[bc]:
      br += 1
    else:
      bc += 1

  if k < dmin: # if U(1) sectors do not match for more than dmin values
    if k == 0: # pathological case with 0 matching colors.
      raise ValueError("No sector matching, M has to be zero")
    s = s[:k]
  s_sort = np.argsort(s)[::-1]
  U = U[:,s_sort]
  s = s[s_sort]
  V = V[s_sort]
  colors = colors[s_sort]
  return U,s,V,colors

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

def move_right(su):
  M_A0 = np.einsum('paurdl,u,d,l->audlpr', su._gammaA, su._lambda_u, su._lambda_d, su._lambda_l)
  M_B0 = np.einsum('paurdl,u,r,d->lpaurd', su._gammaB, su._lambda_d, su._lambda_l, su._lambda_u)
  lambda_dir = su._lambda_r
  D_cst = D**3*d
  D_dir = D
  D_eff = D_dir*d

  M_A0 = M_A0.reshape(D_cst, D_eff)
  colorsMA = [(su._colorsA[1][:,None,None,None] + su._colorsA[2][:,None,None] + su._colorsA[4][:,None] + su._colorsA[5]).ravel(),
              (su._colorsA[0][:,None] + su._colorsA[3]).ravel()]
  W_A, sA, M_A, colors_sA = svdU1(M_A0, colorsMA[0], -colorsMA[1])
  M_A *= sA[:,None]
  print(f"svd A: {lg.norm(W_A @ M_A - M_A0)/lg.norm(M_A0):.1e}")

  M_B0 = M_B0.reshape(D_eff, D_cst)
  colorsMB = [(su._colorsB[5][:,None] + su._colorsB[0]).ravel(), (su._colorsB[1][:,None,None,None]
               + su._colorsB[2][:,None,None] + su._colorsB[3][:,None] + su._colorsB[4]).ravel()]
  M_B, sB, W_B, colors_sB = svdU1(M_B0, colorsMB[0], -colorsMB[1])
  M_B *= sB
  print(f"svd B: {lg.norm(M_B @ W_B - M_B0)/lg.norm(M_B0):.1e}")


  theta = M_A.reshape(D_eff*d, D_dir)
  theta *= su._lambda_r
  theta = np.dot(theta, M_B.reshape(D_dir, d*D_eff) )
  theta = theta.reshape(D_eff, d, d, D_eff).transpose(0,3,1,2).reshape(D_eff**2, d**2)
  theta = np.dot(theta, su._gr)

  theta = theta.reshape(D_eff, D_eff, d, d).swapaxes(1,2).reshape(D_eff*d, D_eff*d)
  row_col = (colors_sA[:,None] + su._colorsA[0]).ravel()
  col_col = (-colors_sB[:,None] + su._colorsB[0]).ravel()
  M_A2,s,M_B2,colors_th = svdU1(theta,row_col,-col_col)
  print(f"svd theta: {lg.norm(M_A2*s @ M_B2 - theta)/lg.norm(theta):.1e}")


  # 4) renormalize link dimension
  s = s[:D_dir]
  s /= s.sum()  # singular values are positive
  su._lambda_r = s
  colors_th = colors_th[:D_dir]

  # 5) start reconstruction of new gammaA and gammaB by unifying cst and eff
  M_A2 = M_A2[:,:D_dir].reshape(D_eff, d*D_dir)
  #colorsMA2 = [colors_sA, (p_col[:,None] - colors_th).ravel()]
  M_A = np.dot(W_A, M_A2)
  colorsMA = [colorsMA[0], (-p_col[:,None] - colors_th).ravel()]
  M_B2 = M_B2[:D_dir].reshape(D_dir*d, D_eff)
  M_B = np.dot(M_B2, W_B)
  colorsMB = [(colors_th + p_col[:,None]).ravel(), colorsMB[1]]

  M_A = M_A.reshape(su._a, su._Du, su._Dd, su._Dl, su._d, su._Dr)
  su._gammaA = np.einsum('audlpr,u,d,l->paurdl', M_A, su._lambda_u**-1, su._lambda_d**-1, su._lambda_l**-1)
  su._colorsA = [-p_col,p_col,v_col,-colors_th,v_col,v_col]
  M_B = M_B.reshape(su._Dr, su._d, su._a, su._Dd, su._Dl, su._Du)
  su._gammaB = np.einsum('lpaurd,u,r,d->paurdl', M_B, su._lambda_d**-1, su._lambda_l**-1, su._lambda_u**-1)
  su._colorsB = [p_col,-p_col,v_col,v_col,v_col,colors_th]


move_right(su)
move_right(su)
