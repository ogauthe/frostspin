import numpy as np
import scipy.linalg as lg
from scipy.sparse.linalg import svds


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

