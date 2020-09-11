import numpy as np
import scipy.linalg as lg
from numba import jit

# cannot use None as default color since -default_color or default_color[:l]
# need not to crash.
default_color = np.array([],dtype=np.int8)

def checkU1(T,colorsT,tol=1e-14):
  """
  Check tensor has U(1) symmetry up to tolerance
  """
  if tuple(len(c) for c in colorsT) != T.shape:
    raise ValueError("Color dimensions do not match tensor")
  for ind in np.array((np.abs(T) > tol).nonzero()).T:
    if sum(colorsT[i][c] for i,c in enumerate(ind)) != 0:
      return ind, T[tuple(ind)]
  return None, 0


def dotU1(a, b, rc_a=default_color, cc_a=default_color, cc_b=default_color, check=False):
  """
  Optimized matrix product for U(1) symmetric matrices. If colors are not
  provided, revert to standard matmul.

  Parameters
  ----------
  a : (m,l) ndarray
    First argument.
  b : (l,n) ndarray
    Second argument.
  rc_a : (m,) integer ndarray
    U(1) quantum numbers of a rows.
  cc_a : (l,) integer ndarray
    U(1) quantum numbers of a columns. Must be the opposite of b row colors.
  cc_b : (n,) integer ndarray
    U(1) quantum numbers of b columns.
  check : bool (debug only)
    Wether to check results by displaying norm(dotU1-np.dot)/norm(dot).

  Returns
  -------
  output : (m,n) ndarray
    dot product of a and b.
  """
  return np.dot(a,b)
  # revert to standard matmul if colors are missing
  if not rc_a.size or not cc_a.size or not cc_b.size:
    if check:
      print('dotU1 revert to np.__matmul__')
    return a @ b

  if a.ndim == b.ndim != 2:
    raise ValueError("ndim must be 2 to use dot")
  if a.shape[0] != rc_a.shape[0]:
    raise ValueError('a rows and row colors shape mismatch')
  if a.shape[1] != cc_a.shape[0]:
    raise ValueError('a columns and column colors shape mismatch')
  if b.shape[1] != cc_b.shape[0]:
    raise ValueError('b columns and column colors shape mismatch')
  if a.shape[1] != b.shape[0]:
    raise ValueError("shape mismatch between a columns and b rows")

  res = np.zeros((a.shape[0], b.shape[1]))
  for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
    ri = (rc_a == -c).nonzero()[0][:,None]
    mid = (cc_a == c).nonzero()[0]
    ci = cc_b == c
    res[ri,ci] = a[ri,mid] @ b[mid[:,None],ci]
  if check:
    ex = a @ b
    print(f"dotU1: \033[33m{lg.norm(res-ex)/lg.norm(ex):.1e}\033[0m")
  return res


def combine_colors(*colors):
  """
  Construct colors of merged tensor legs from every leg colors.
  """
  if not colors[0].size:
    return default_color
  combined = colors[0]
  for c in colors[1:]:
    combined = (combined[:,None]+c).ravel()
  return combined


def tensordotU1(a, b, ax_a, ax_b, colors_a=None, colors_b=None, check=False):
  """
  Optimized tensor dot product along specified axes for U(1) symmetric tensors.
  If colors are not provided, revert to numpy tensordot.

  Parameters
  ----------
  a,b : ndarray
    tensors to contract.
  ax_a, ax_b : tuple of integers.
    Axes to contract for tensors a and b.
  colors_a, colors_b : list of a.ndim and b.ndim integer arrays.
    U(1) quantum numbers of a and b axes.
  check : bool (debug only)
    Wether to check results by displaying norm(tensordotU1-np.tensordot)

  Returns
  -------
  output : ndarray
    Tensor dot product of a and b.

  """
  return np.tensordot(a, b, (ax_a, ax_b))
  # call np.tensordot if colors are not provided
  if colors_a is None or colors_b is None or not colors_a[0].size or not colors_b[0].size:
    if check:
      print('tensordotU1 reverted to np.tensordot')
    return np.tensordot(a, b, (ax_a, ax_b))

  if len(ax_a) != len(ax_b):
    raise ValueError("axes for a and b must match")
  if len(ax_a) > a.ndim:
    raise ValueError("axes for a do not match array")
  if len(ax_b) > b.ndim:
    raise ValueError("axes for b do not match array")
  dim_contract = tuple(a.shape[ax] for ax in ax_a)
  if dim_contract != tuple(b.shape[ax] for ax in ax_b):
    raise ValueError("dimensions for a and b do not match")

  # copy np.tensordot
  notin_a = tuple(k for k in range(a.ndim) if k not in ax_a) # free leg indices
  notin_b = tuple([k for k in range(b.ndim) if k not in ax_b])
  dim_free_a = tuple(a.shape[ax] for ax in notin_a)
  dim_free_b = tuple(b.shape[ax] for ax in notin_b)

  # construct merged colors of a free legs, contracted legs and b free legs
  rc_a = combine_colors(*[colors_a[ax] for ax in notin_a])
  cc_a = combine_colors(*[colors_a[ax] for ax in ax_a])
  cc_b = combine_colors(*[colors_b[ax] for ax in notin_b])

  # np.tensordot algorithm transposes a and b to put contracted axes
  # at end of a and begining of b, reshapes to matrices and compute matrix prod
  # here avoid complete construction of at and bt which requires copy
  # compute indices of relevant coeff (depending on colors) of at and bt, then
  # transform them into (flat) indices of a and b. Copy only relevant blocks
  # into small matrices and compute dot product
  # (cannot compute directly indices of a and b because of merged legs)
  sh_at = dim_free_a + dim_contract   # shape of a.transpose(free_a + ax_a)
  prod_a = np.prod(dim_contract)      # offset of free at indices
  div_a = np.array([*sh_at,1])[:0:-1].cumprod()[::-1] # 1D index -> multi-index
  # multi-index of at -> 1D index of a by product with transposed shape
  cp_a = np.array([*a.shape,1])[:0:-1].cumprod()[::-1][np.array(notin_a+ax_a)]

  prod_b = np.prod(dim_free_b)
  sh_bt = dim_contract + dim_free_b
  div_b = np.array([*sh_bt,1])[:0:-1].cumprod()[::-1]
  cp_b = np.array([*b.shape,1])[:0:-1].cumprod()[::-1][np.array(ax_b+notin_b)]

  res = np.zeros(dim_free_a + dim_free_b)
  for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
    ri = (rc_a == -c).nonzero()[0][:,None]
    mid = (cc_a == c).nonzero()[0]
    ci = (cc_b == c).nonzero()[0]
    ind_a = ((ri*prod_a + mid)[:,:,None]//div_a%sh_at) @ cp_a
    ind_b = ((mid[:,None]*prod_b + ci)[:,:,None]//div_b%sh_bt) @ cp_b
    res.flat[ri*prod_b + ci] = a.flat[ind_a] @ b.flat[ind_b]
  if check:
    ex = np.tensordot(a,b,(ax_a,ax_b))
    print(f"tensordotU1: \033[33m{lg.norm(res-ex)/lg.norm(ex):.1e}\033[0m")
  return res


@jit(nopython=True)
def svdU1(M, row_colors=default_color, col_colors=default_color, check=False):
  """
  Singular value decomposition for a U(1) symmetric matrix M. Revert to
  standard svd if colors are not provided.

  Parameters
  ----------
  M : (m,n) ndarray
    Matrix to decompose.
  row_colors : (m,) integer ndarray
    U(1) quantum numbers of the rows.
  col_colors : (n,) integer ndarray
    U(1) quantum numbers of the columns.
  check : bool (debug only)
    Wether to check results by display norm(U@s@V-M)/norm(s)

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

  # revert to standard svd if colors are not provided
  if not row_colors.size or not col_colors.size:
    if check:
      print("no color provided, svdU1 reverted to np.linalg.svd")
    U,s,V = np.linalg.svd(M, full_matrices=False)
    return U,s,V,default_color

  if M.ndim != 2:
    raise ValueError("M has to be a matrix")
  if row_colors.shape != (M.shape[0],):
    raise ValueError("row_colors has to be (M.shape[0])")
  if col_colors.shape != (M.shape[1],):
    raise ValueError("col_colors has to be (M.shape[1])")

  row_sort = row_colors.argsort()
  sorted_row_colors = row_colors[row_sort]
  col_sort = col_colors.argsort()
  sorted_col_colors = col_colors[col_sort]
  row_inds = [0] + list((sorted_row_colors[:-1] != sorted_row_colors[1:]
                    ).nonzero()[0] + 1) + [M.shape[0]]
  col_inds = [0] + list((sorted_col_colors[:-1] != sorted_col_colors[1:]
                    ).nonzero()[0] + 1) + [M.shape[1]]
  dmin = min(M.shape)
  U = np.zeros((M.shape[0],dmin))
  s = np.empty(dmin)
  V = np.zeros((dmin,M.shape[1]))
  colors = np.empty(dmin,dtype=np.int8)

  # match blocks with same color and compute SVD inside those blocks only
  k,br,bc,brmax,bcmax = 0,0,0,len(row_inds)-1,len(col_inds)-1
  while br < brmax and bc < bcmax:
    if sorted_row_colors[row_inds[br]] == sorted_col_colors[col_inds[bc]]:
      ir = row_sort[row_inds[br]:row_inds[br+1]]
      ic = col_sort[col_inds[bc]:col_inds[bc+1]]
      m = np.ascontiguousarray(M[ir][:,ic])
      d = min(m.shape)
      U[ir,k:k+d], s[k:k+d], V[k:k+d:,ic] = np.linalg.svd(m,full_matrices=False)
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
  s_sort = s.argsort()[::-1]
  U = U[:,s_sort]
  s = s[s_sort]
  V = V[s_sort]
  colors = colors[s_sort]
  if check:
    r = np.linalg.norm(U*s@V-M)/np.linalg.norm(M)
    print("svdU1:\033[33m", r, "\033[0m")
  return U,s,V,colors

