import numpy as np
import scipy.linalg as lg


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


def checkU1(T,colorsT,tol=1e-14):
  nn = np.transpose((np.abs(T) > 1e-16).nonzero())
  for ind in nn:
    if sum(colorsT[i][c] for i,c in enumerate(ind)) != 0:
      return False, ind, T[tuple(ind)]
  return True, None, 0


def tensordotU1(a, colors_a, ax_a, b, colors_b, ax_b):
  if len(ax_a) != len(ax_b):
    raise ValueError("axes for a and b must match")
  if len(ax_a) > a.ndim:
    raise ValueError("axes for a do not match array")
  if len(ax_b) > b.ndim:
    raise ValueError("axes for b do not match array")
  dim_contract = tuple(a.shape[ax] for ax in ax_a)
  if dim_contract != tuple(b.shape[ax] for ax in ax_b):
    raise ValueError("dimensions for a and b do not match")

  notin_a = tuple(k for k in range(a.ndim) if k not in ax_a)  # free leg indices
  notin_b = tuple([k for k in range(b.ndim) if k not in ax_b])  # free leg indices
  dim_free_a = tuple(a.shape[ax] for ax in notin_a)
  dim_free_b = tuple(b.shape[ax] for ax in notin_b)
  cp_a = np.array([*a.shape,1])[:0:-1].cumprod()[::-1][np.array(notin_a+ax_a)]
  cp_b = np.array([*a.shape,1])[:0:-1].cumprod()[::-1][np.array(ax_b+notin_b)]

  rc_a = combine_colors(*[colors_a[ax] for ax in notin_a])
  cc_a = combine_colors(*[colors_a[ax] for ax in ax_a])
  cc_b = combine_colors(*[colors_b[ax] for ax in notin_b])

  # avoid construction of at and bt = b.transpose(ax_b + notin_b)
  sh_at = dim_free_a + dim_contract
  prod_a = np.prod(dim_contract)
  div_a = np.array([*sh_at,1])[:0:-1].cumprod()[::-1]

  prod_b = np.prod(dim_free_b)
  sh_bt = dim_contract + dim_free_b
  div_b = np.array([*sh_bt,1])[:0:-1].cumprod()[::-1]

  res = np.zeros(dim_free_a + dim_free_b)
  for c in set(-rc_a).intersection(set(cc_a)).intersection(set(cc_b)):
    ri = (rc_a == -c).nonzero()[0][:,None]
    mid = (cc_a == c).nonzero()[0]
    ci = (cc_b == c).nonzero()[0]
    ind_a = ((ri*prod_a + mid)[:,:,None]//div_a%sh_at) @ cp_a
    ind_b = ((mid[:,None]*prod_b + ci)[:,:,None]//div_b%sh_bt) @ cp_b
    res.flat[ri*prod_b + ci] = a.flat[ind_a] @ b.flat[ind_b]
  return res



# obligé de repasser en format 1D transposé pour combiner les indices ri et mid
# transposition nécessaire pour traiter séparément à gauche ri et à droite contracted
# ensuite repasse aux indices de a en multipliant par les dimensions transposées à l'inverse

# générer directement indices 1D valables de a en multipliant ri et mid par ce qui va bien
# plus besoin de repasser en multi-indice pr transposer
# donner directement à a.flat
# Q : est-ce que liste 1D obtenue nécessite transposition avant reshape matrice ?
