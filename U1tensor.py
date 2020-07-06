import numpy as np


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

  def reshape(self,sh, nc)
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

def dotU1(a, rc_a, cc_a, b, cc_b):
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
  combined = np.zeros(1,dtype=np.int8)
  for c in colors:
    combined = (combined[:,None]+c).reshape(len(combined)*len(c))
  return combined


def tensordotU1(a, colors_a, ax_a, b, colors_b, ax_b):
  if len(ax_a) != len(ax_b):
    raise ValueError("axes for a and b must match")
  if len(ax_a) > a.ndim:
    raise ValueError("axes for a do not match array")
  if len(ax_b) > b.ndim:
    raise ValueError("axes for b do not match array")

  notin_a = tuple(k for k in range(a.ndim) if k not in ax_a)  # free leg indices
  free_a = tuple(a.shape[ax] for ax in notin_a)
  contract = tuple(a.shape[ax] for ax in ax_a)
  at = a.transpose(notin_a + ax_a)
  cumprod_a = np.array([*free_a,1])[:0:-1].cumprod()[::-1]
  cumprod_mid = np.array([*contract,1])[:0:-1].cumprod()[::-1]

  notin_b = [k for k in range(b.ndim) if k not in ax_b]  # free leg indices
  free_b = tuple(b.shape[ax] for ax in notin_b)
  bt = b.transpose(ax_b + notin_b)
  cumprod_b = np.array([*free_b,1])[:0:-1].cumprod()[::-1]

  rc_a = combine_colors(*[colors_a[i] for i in notin_a])
  cc_a = combine_colors(*[colors_a[i] for i in ax_a])
  cc_b = combine_colors(*[colors_b[i] for i in notin_b])

  res = np.zeros((np.prod(free_a), np.prod(free_b)))
  for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
    ri = (rc_a == -c).nonzero()[0][:,None]//cumprod_a%free_a
    mid = (cc_a == c).nonzero()[0][:,None]//cumprod_mid%contract
    ci = (cc_b == c).nonzero()[0][:,None]//cumprod_b%free_b
    ind_at = np.ix_(*ri,*mid)
    ind_bt = np.ix_(*mid,*ci)
    res[ri,ci] = at[ind_at].reshape(len(ri),len(mid)) @ bt[ind_b].reshape(len(mid),len(ci))
  return res.reshape(free_a + free_b)
