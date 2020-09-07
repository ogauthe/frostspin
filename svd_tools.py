import numpy as np
import scipy.linalg as lg
from scipy.sparse.linalg import eigsh  # use custom svds
from toolsU1 import default_color
from scipy.sparse.linalg.interface import LinearOperator


# use custom svds adapted from scipy to remove small value cutoff
# remove some other unused features: impose which='LM' and solver='arpack'
# always sort values by decreasing order
# compute either no singular vectors or both U and V
# ref: https://github.com/scipy/scipy/pull/11829
def svds(A, k=6, ncv=None, tol=0, v0=None,
         maxiter=None, return_singular_vectors=True):

  A = np.asarray(A)  # do not consider LinearOperator or sparse matrix
  n, m = A.shape
  dmin = min(m,n)

  if k <= 0 or k >= dmin:
    raise ValueError("k must be between 1 and min(A.shape), k=%d" % k)
  if n > m:
    X_dot = X_matmat = A.dot
    XH_dot = XH_mat = A.T.conj().dot
    transpose = False
  else:
    XH_dot = XH_mat = A.dot
    X_dot = X_matmat = A.T.conj().dot
    transpose = True

  def matvec_XH_X(x):
    return XH_dot(X_dot(x))

  def matmat_XH_X(x):
    return XH_mat(X_matmat(x))

  XH_X = LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                        matmat=matmat_XH_X,
                        shape=(dmin, dmin))

  eigvals, eigvec = eigsh(XH_X, k=k, tol=tol, maxiter=maxiter,
                          ncv=ncv, which='LM', v0=v0)
  u = X_matmat(eigvec)
  if not return_singular_vectors:
    s = svd(u, compute_uv=False)
    return s

  # compute the right singular vectors of X and update the left ones accordingly
  u, s, vh = lg.svd(u, full_matrices=False)
  if transpose:
    u, vh = eigvec @ vh.T.conj(), u.T.conj()
  else:
    vh = vh @ eigvec.T.conj()
  return u, s, vh




def svd_truncate(M, chi, keep_multiplets=False, window=10, cuttol=1e-6,
                 maxiter=1000):
  """
  Compute a given number of singular values and singular vectors.

  Parameters
  ----------
  M : (m,n) ndarray
    Matrix to decompose.
  chi : integer
    Number of singular vectors to compute. Effective cut depends on
    keep_multiplets (see notes)
  keep_multiplets : bool
    If true, compute more than chi values and cut between two different
    multiplets
  window : integer
    If keep_multiplets is True, compute chi+window vectors and cut between chi
    and chi+window
  cuttol : float
    Tolerance to consider two consecutive values as degenerate.
  maxiter : integer
    Maximal number of iterations. Finite allows clean crash instead of running
    forever.

  Returns
  -------
  U :   U : (m,cut) ndarray
    Left singular vectors.
  s : (cut,) ndarray
    Singular values.
  V : (cut,n) right singular vectors

  Notes:
  ------
  cut is fixed as chi if keep_multiplets is False, else as the smallest value
  in [chi,chi+window[ such that s[cut+1] < cuttol*s[cut]
  """
  if keep_multiplets:
    U,s,V = svds(M, k=chi+window, maxiter=maxiter)
    cut = chi + (s[chi:] < cuttol*s[chi-1:-1]).nonzero()[0][0]
  else:
    U,s,V = svds(M, k=chi, maxiter=maxiter)
    cut = chi

  cut = min(cut,(s>0).nonzero()[0][-1]+1)  # remove exact zeros
  U,s,V = U[:,:cut], s[:cut], V[:cut]
  return U,s,V


def svdU1_truncate(M, chi, row_colors=default_color, col_colors=default_color,
                  keep_multiplets=False, window=10, cuttol=1e-6, maxiter=1000):
  """
  Compute a given number of singular values and singular vectors using U(1)
  symmetry.

  Parameters
  ----------
  M : (m,n) ndarray
    Matrix to decompose.
  chi : integer
    Number of singular vectors to compute. Effective cut depends on
    keep_multiplets (see notes)
  row_colors : (m,) integer ndarray
    U(1) quantum numbers of the rows.
  col_colors : (n,) integer ndarray
    U(1) quantum numbers of the columns.
  keep_multiplets : bool
    If true, compute more than chi values and cut between two different
    multiplets
  window : integer
    If keep_multiplets is True, compute chi+window vectors and cut between chi
    and chi+window
  cuttol : float
    Tolerance to consider two consecutive values as degenerate.
  maxiter : integer
    Maximal number of iterations. Finite allows clean crash instead of running
    forever.

  Returns
  -------
  U :   U : (m,cut) ndarray
    Left singular vectors.
  s : (cut,) ndarray
    Singular values.
  V : (cut,n) right singular vectors
  color : (cut,) ndarray with int8 data type
    U(1) quantum numbers of s.

  Notes:
  ------
  cut is fixed as chi if keep_multiplets is False, else as the smallest value
  in [chi,chi+window[ such that s[cut+1] < cuttol*s[cut].
  It is assumed that degenerate values belong to separate U(1) sectors. This is
  True for SU(2) symmetry, NOT for SU(N>2).
  """

  # revert to svd_truncate if no color is given
  if row_colors is None or not row_colors.size or col_colors is None or not col_colors.size:
    U,s,V = svd_truncate(M, chi, keep_multiplets, window, cuttol, maxiter)
    return U,s,V,default_color

  row_sort = row_colors.argsort()
  sorted_row_colors = row_colors[row_sort]
  col_sort = col_colors.argsort()
  sorted_col_colors = col_colors[col_sort]
  row_inds = np.array([0, *((sorted_row_colors[:-1] != sorted_row_colors[1:]
                    ).nonzero()[0] + 1), M.shape[0]])
  col_inds = [0, *((sorted_col_colors[:-1] != sorted_col_colors[1:]
                    ).nonzero()[0] + 1), M.shape[1]]

  # we need to compute chi singular values in every sector to deal with the
  # worst case every chi largest values in the same sector.
  # A fully memory efficient code would only store 2*chi vectors. Here we
  # select the chi largest values selection once all vector are computed, so we
  # need to store sum( min(chi, sector_size) for each sector) different vectors.
  # Dealing with different block numbers in row_inds and col_inds here is
  # cumbersome, so we consider the sub-optimal min(chi, sector_rows_number)
  # while columns number may be smaller.
  # in CTMRG projector construction, row and column colors are the same anyway.
  max_k = np.minimum(chi,row_inds[1:] - row_inds[:-1]).sum()
  U = np.zeros((M.shape[0],max_k))
  S = np.empty(max_k)
  V = np.zeros((max_k,M.shape[1]))
  colors = np.empty(max_k,dtype=np.int8)

  # match blocks with same color and compute SVD inside those blocks only
  k,br,bc,brmax,bcmax = 0,0,0,row_inds.shape[0]-1,len(col_inds)-1
  while br < brmax and bc < bcmax:
    if sorted_row_colors[row_inds[br]] == sorted_col_colors[col_inds[bc]]:
      ir = row_inds[br]
      jr = row_inds[br+1]
      ic = col_inds[bc]
      jc = col_inds[bc+1]
      m = np.ascontiguousarray(M[row_sort[ir:jr,None], col_sort[ic:jc]])
      if min(m.shape) < 3*chi:  # use exact svd for small blocks
        u,s,v = lg.svd(m, full_matrices=False)
      else:
        u,s,v = svds(m, k=chi, maxiter=maxiter)

      d = min(chi,s.shape[0])   # may be smaller than chi
      U[row_sort[ir:jr],k:k+d] = u[:,:d]
      S[k:k+d] = s[:d]
      V[k:k+d,col_sort[ic:jc]] = v[:d]
      colors[k:k+d] = sorted_row_colors[row_inds[br]]
      k += d
      br += 1
      bc += 1
    elif sorted_row_colors[br] < sorted_col_colors[bc]:
      br += 1
    else:
      bc += 1

  S = S[:k]  # k <= max_k
  s_sort = S.argsort()[::-1]
  S = S[s_sort]

  # expect multiplets to lie in separate color blocks (true for SU(2) > U(1))
  if keep_multiplets:
    cut = chi + (S[chi:] < cuttol*S[chi-1:-1]).nonzero()[0][0]
  else:
    cut = chi

  cut = min(cut,(S>0).nonzero()[0][-1]+1)  # remove exact zeros
  U,S,V,colors = U[:,s_sort[:cut]], S[:cut], V[s_sort[:cut]], colors[s_sort[:cut]]
  return U,S,V,colors
