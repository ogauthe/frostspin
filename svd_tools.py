import numpy as np
import scipy.linalg as lg
from scipy.sparse.linalg import svds
from toolsU1 import default_color, svdU1


def svd_truncate(M, chi, keep_multiplets=False, window=10, cuttol=1e-6, maxiter=1000):
  if keep_multiplets:
    U,s,V = svds(M, k=chi+window, maxiter=maxiter)
    cut = chi + (s[chi:] < cuttol*s[chi-1:-1]).nonzero()[0][0]
  else:
    U,s,V = svds(M, k=chi, maxiter=maxiter)
    cut = chi

  cut = min(cut,(s>0).nonzero()[0][-1]+1)  # remove exact zeros
  U,s,V = U[:,:cut], s[:cut], V[:cut]
  return U,s,V

def svdU1_truncate(M, chi, row_colors=default_color, col_colors=default_color, keep_multiplets=False, window=10, cuttol=1e-6, maxiter=1000):

  # revert to svd_truncate if no color is given
  if not row_colors.size or not col_colors.size:
    U,s,V = svd_truncate(M, chi, keep_multiplets, window, cuttol, maxiter)
    return U,s,V,default_color

  row_sort = np.argsort(row_colors)
  sorted_row_colors = row_colors[row_sort]
  col_sort = np.argsort(col_colors)
  sorted_col_colors = col_colors[col_sort]
  row_inds = np.array([0, *((sorted_row_colors[:-1] != sorted_row_colors[1:]
                    ).nonzero()[0] + 1), M.shape[0]])
  col_inds = [0, *((sorted_col_colors[:-1] != sorted_col_colors[1:]
                    ).nonzero()[0] + 1), M.shape[1]]

  # we need to compute at most chi singular values in every sector
  # set an upper bond of number of vectors to compute
  # this could be less if some blocks are smaller or absent in col_inds
  # but dealing with different sizes row_inds and col_inds is cumbersome
  # (doing it imposes keeping only colors appearing on both rows and columns)
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
  s_sort = np.argsort(S)[::-1]
  S = S[s_sort]

  # expect multiplets to lie in separate color blocks (true for SU(2) downgraded to U(1))
  if keep_multiplets:
    cut = chi + (S[chi:] < cuttol*S[chi-1:-1]).nonzero()[0][0]
  else:
    cut = chi

  cut = min(cut,(S>0).nonzero()[0][-1]+1)  # remove exact zeros
  U,S,V,colors = U[:,s_sort[:cut]], S[:cut], V[s_sort[:cut]], colors[s_sort[:cut]]
  return U,S,V,colors
