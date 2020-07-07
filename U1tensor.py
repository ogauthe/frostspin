import numpy as np
import scipy.linalg as lg
from toolsU1 import checkU1, tensordotU1


class U1tensor(object):

  def __init__(self, ar, colors, check_axes=True):
    if ar.ndim != len(colors):
      raise ValueError("array and colors must have same length")
    if check_axes:
      for i,d in enumerate(ar.shape):
        if len(colors[i]) != d:
          raise ValueError("array and colors must have same dim on every axis")
    self._ar = ar
    self._colors = colors

  def ar(self):
    return self._ar

  def shape(self):
    return self._ar.shape

  def ndim(self):
    return self._ar.ndim

  # clever thing would be to encode view for tensor: .T, transpose or swapaxes
  # would encode views with views of colors and array
  def swapaxes(self,ax1,ax2):
    colors = self.colors.copy()
    colors[ax1] = self.colors[ax2]
    colors[ax2] = self.colors[ax1]
    return U1tensor(self._ar.swapaxes(ax1,ax2), colors, False)

  def transpose(self,axes):
    colors = [self._colors[i] for i in axes]
    return U1tensor(self._ar.transpose(axes), colors, False)

  def reshape(self,sh, nc):
    return U1tensor(self._ar.reshape(sh),nc)

  def copy(self):
    return U1tensor(self._ar.copy(), self._colors.copy(), False)

  def dot(self,other):
    if self._ar.ndim == other._ar.ndim != 2:
      raise ValueError("ndim must be 2 to use dot")
    res = np.zeros((self._ar.shape[0],other._ar.shape[1]))
    for c in set(self._colors[0]):
      ri = (self._colors[0] == c).nonzero()[0][:,None]
      ci = other._colors[1] == c
      mid = (self._colors[1]==-c).nonzero()[0]
      assert( ((other._colors[0] == c) == mid).all()), "Colors do not match"
      res[ri,ci] = self._ar[ri,mid] @ other._ar[mid[:,None],ci]
    return res

  def tensordot(self,b,ax,ax_b):
    ar = tensordotU1(self._ar, self._colors, ax, b._ar, b._colors, ax_b)
    cols = [self._colors[k] for k in range(self._ar.ndim) if k not in ax]\
           + [b._colors[k] for k in range(b._ar.ndim) if k not in ax_b]
    return U1tensor(ar, cols, False)

  def checkU1(self):
    return checkU1(self._ar, self._colors, tol)


