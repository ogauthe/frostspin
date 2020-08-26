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

def svdU1_truncate(M, chi, rc=default_color, cc=default_color, keep_multiplets=False, window=10, cuttol=1e-6, maxiter=1000):
  if not rc.size or not cc.size:
    U,s,V = svd_truncate(M, chi, keep_multiplets, window, cuttol, maxiter)
    return U,s,V,default_color

  U,s,V,col = svdU1(M,rc,cc)
  if keep_multiplets:
    cut = chi + (s[chi:] < cuttol*s[chi-1:-1]).nonzero()[0][0]
  else:
    cut = chi

  cut = min(cut,(s>0).nonzero()[0][-1]+1)  # remove exact zeros
  U,s,V,col = U[:,:cut], s[:cut], V[:cut], col[:cut]
  return U,s,V,col
