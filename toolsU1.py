import numpy as np
import scipy.linalg as lg


def checkU1(T,colorsT,tol=1e-14):
  """
  Check tensor has U(1) symmetry up to tolerance
  """
  for ind in np.array((np.abs(T) > tol).nonzero()).T:
    if sum(colorsT[i][c] for i,c in enumerate(ind)) != 0:
      return False, ind, T[tuple(ind)]
  return True, None, 0


def dotU1(a, rc_a, cc_a, b, cc_b):
  """
  Optimized matrix product for U(1) symmetric matrices.

  Parameters
  ----------
  a : (m,l) ndarray
    First argument.
  rc_a : (m,) integer ndarray
    U(1) quantum numbers of a rows.
  cc_a : (l,) integer ndarray
    U(1) quantum numbers of a columns. Must be the opposite of b row colors.
  b : (l,n) ndarray
    Second argument.
  cc_b : (n,) integer ndarray
    U(1) quantum numbers of b columns.

  Returns
  -------
  output : (m,n) ndarray
    dot product of a and b.
  """
  if a.ndim == b.ndim != 2:
    raise ValueError("ndim must be 2 to use dot")
  if a.shape[1] != b.shape[0]:
    raise ValueError("a col number must be equal to a rows")

  res = np.zeros((a.shape[0], b.shape[1]))
  for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
    ri = (rc_a == -c).nonzero()[0][:,None]
    mid = (cc_a == c).nonzero()[0]
    ci = cc_b == c
    res[ri,ci] = a[ri,mid] @ b[mid[:,None],ci]
  return res


def combine_colors(*colors):
  """
  Construct colors of merged tensor legs from every leg colors.
  """
  combined = colors[0]
  for c in colors[1:]:
    combined = (combined[:,None]+c).ravel()
  return combined


def tensordotU1(a, colors_a, ax_a, b, colors_b, ax_b):
  """
  Optimized tensor dot product along specified axes for U(1) symmetric tensors.

  Parameters
  ----------
  a,b : ndarray
    tensors to contract.
  colors_a, colors_b : list of a.ndim and b.ndim integer arrays.
    U(1) quantum numbers of a and b axes.
  ax_a, ax_b : tuple of integers
    axes to contract for tensors a and b

  Returns
  -------
  output : ndarray
    Tensor dot product of a and b.

  """
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
  cp_b = np.array([*a.shape,1])[:0:-1].cumprod()[::-1][np.array(ax_b+notin_b)]

  res = np.zeros(dim_free_a + dim_free_b)
  for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
    ri = (rc_a == -c).nonzero()[0][:,None]
    mid = (cc_a == c).nonzero()[0]
    ci = (cc_b == c).nonzero()[0]
    ind_a = ((ri*prod_a + mid)[:,:,None]//div_a%sh_at) @ cp_a
    ind_b = ((mid[:,None]*prod_b + ci)[:,:,None]//div_b%sh_bt) @ cp_b
    res.flat[ri*prod_b + ci] = a.flat[ind_a] @ b.flat[ind_b]
  return res


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

