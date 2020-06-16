import numpy as np
import scipy.linalg as lg

def svd_SU(M_A, M_B, lambda_dir, gate, D_cst, d, D_dir):
  """
  utilitary function
  Construct matrix theta from tensors M_A and M_B, where lambda have been added
  to gammaA and gammaB. Cut it using SVD.
  """

  D_eff = D_dir*d

  # 1) SVD cut between constant tensors and effective tensor to update
  # hence reduce main SVD to dimension D_eff*d < D_cst*d
  #     \|        \|
  #     -A-    -> -W==M-
  #      |\        |   \
  M_A = M_A.reshape(D_cst, D_eff)
  W_A, sA, M_A = lg.svd(M_A, full_matrices=False)
  M_B = M_B.reshape(D_eff, D_cst)
  M_B, sB, W_B = lg.svd(M_B, full_matrices=False)

  # 2) construct matrix theta with gate g
  #
  #             =MA-lr-MB=
  #                \  /
  #   theta =       gg
  #                /  \
  theta = M_A.reshape(D_eff*d, D_dir)
  theta = np.einsum('ij,j->ij', theta, lambda_dir)
  theta = np.dot(theta, M_B.reshape(D_dir, d*D_eff) )
  theta = theta.reshape(D_eff, d, d, D_eff).transpose(0,3,1,2).reshape(D_eff**2, d**2)
  theta = np.dot(theta, gate)

  # 3) cut theta with SVD
  theta = theta.reshape(D_eff, D_eff, d, d).swapaxes(1,2).reshape(D_eff*d, D_eff*d)
  M_A,s,M_B = lg.svd(theta)

  # 4) renormalize link dimension
  s = s[:D_dir]
  s /= s.sum()  # singular values are positive

  # 5) start reconstruction of new gammaA and gammaB by unifying cst and eff
  M_A = M_A[:,:D_dir].reshape(D_eff, d*D_dir)
  M_A = np.einsum('i,ij->ij', sA, M_A)
  M_A = np.dot(W_A, M_A)
  M_B = M_B[:D_dir].reshape(D_dir*d, D_eff)
  M_B = np.einsum('ij,j->ij', M_B, sB)
  M_B = np.dot(M_B, W_B)
  return M_A, s, M_B


class SimpleUpdateAB(object):

  def __init__(self, sh, gates, A0=None, B0=None):
    """
    sh: tuple of int, shape of tensor A.
    form (d,a,Du,Dr,Dd,Dl), where a=1 for pure wavefunction and a=d for thermal TN
    shape of B is then (d,a,Dd,Dl,Du,Dr)
    """

    self._d, self._a, self._Du, self._Dr, self._Dd, self._Dl = sh

    if self._Du*self._d > self._a*self._Dr*self._Dd*self._Dl:
      raise ValueError('up bond: D_eff > D_cst, cannot reshape')
    if self._Dr*self._d > self._a*self._Du*self._Dd*self._Dl:
      raise ValueError('right bond: D_eff > D_cst, cannot reshape')
    if self._Dd*self._d > self._a*self._Du*self._Dr*self._Dl:
      raise ValueError('down bond: D_eff > D_cst, cannot reshape')
    if self._Dl*self._d > self._a*self._Du*self._Dr*self._Dd:
      raise ValueError('left bond: D_eff > D_cst, cannot reshape')

    if A0 is not None:
      if A0.shape is not sh:
        raise ValueError("invalid shape for A0")
    else:
      A0 = np.random.random(sh)
    if B0 is not None:
      if B0.shape is not (self._d,self._a,self._Dd,self._Dl,self._Du,self._Dr):
        raise ValueError("invalid shape for B0")
    else:
      B0 = np.random.random((self._d,self._a,self._Dd,self._Dl,self._Du,self._Dr))

    self._gu, self._gr, self._gd, self._gl = gates
    shg = (self._d**2, self._d**2)
    if self._gu.shape != shg:
      raise ValueError('invalid shape for up gate')
    if self._gr.shape != shg:
      raise ValueError('invalid shape for right gate')
    if self._gd.shape != shg:
      raise ValueError('invalid shape for down gate')
    if self._gl.shape != shg:
      raise ValueError('invalid shape for left gate')

    self._gammaA = A0
    self._gammaB = B0
    # using structure to store lambdas just makes code unclear and not much more versatile.
    self._lambda_u = np.ones(self._Du)
    self._lambda_r = np.ones(self._Dr)
    self._lambda_d = np.ones(self._Dd)
    self._lambda_l = np.ones(self._Dl)


  def get_AB(self):
    """
    return optimized tensors A and B.
    A and B are obtained by adding relevant sqrt(lambda) to every leg of gammaA
    and gammaB
    """
    u = np.sqrt(self._lambda_u)
    r = np.sqrt(self._lambda_r)
    d = np.sqrt(self._lambda_d)
    l = np.sqrt(self._lambda_l)
    A = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaA,u,r,d,l)
    B = np.einsum('paurdl,u,r,d,l->paurdl',self._gammaB,d,l,u,r)
    A /= lg.norm(A)
    B /= lg.norm(B)
    return A,B


  def update(self):
    """
    update all links
    """
    # TODO: do not add and remove lambdas every time, keep some
    self.update_up()
    self.update_right()
    self.update_down()
    self.update_left()
    # 2nd order Trotter by reversing order
    self.update_left()
    self.update_down()
    self.update_right()
    self.update_up()

    self._gammaA /= lg.norm(self._gammaA)
    self._gammaB /= lg.norm(self._gammaB)

  def update_right(self):
    """
    update lambda_r by applying gate gr to right link
    """
    # add diagonal lambdas to gammaA and gammaB
    M_A = np.einsum('paurdl,u,d,l->audlpr', self._gammaA, self._lambda_u, self._lambda_d, self._lambda_l)
    M_B = np.einsum('paurdl,u,r,d->lpaurd', self._gammaB, self._lambda_d, self._lambda_l, self._lambda_u)

    # construct matrix theta, renormalize bond dimension and get back tensors
    M_A, self._lambda_r, M_B = svd_SU(M_A, M_B, self._lambda_r, self._gr, self._a*self._Du*self._Dd*self._Dl, self._d, self._Dr)

    # define new gammaA and gammaB from renormalized M_A and M_B
    M_A = M_A.reshape(self._a, self._Du, self._Dd, self._Dl, self._d, self._Dr)
    self._gammaA = np.einsum('audlpr,u,d,l->paurdl', M_A, self._lambda_u**-1, self._lambda_d**-1, self._lambda_l**-1)
    M_B = M_B.reshape(self._Dr, self._d, self._a, self._Dd, self._Dl, self._Du)
    self._gammaB = np.einsum('lpaurd,u,r,d->paurdl', M_B, self._lambda_d**-1, self._lambda_l**-1, self._lambda_u**-1)


  def update_down(self):
    """
    update lambda_d by applying gate gd to down bond
    only differences from update_right are leg positions
    """
    M_A = np.einsum('paurdl,u,r,l->aurlpd', self._gammaA, self._lambda_u, self._lambda_r, self._lambda_l)
    M_B = np.einsum('paurdl,r,d,l->upardl', self._gammaB, self._lambda_l, self._lambda_u, self._lambda_r)
    M_A, self._lambda_d, M_B = svd_SU(M_A, M_B, self._lambda_d, self._gd, self._a*self._Du*self._Dr*self._Dl, self._d, self._Dd)
    M_A = M_A.reshape(self._a, self._Du, self._Dr, self._Dl, self._d, self._Dd)
    self._gammaA = np.einsum('aurlpd,u,r,l->paurdl', M_A, self._lambda_u**-1, self._lambda_r**-1, self._lambda_l**-1)
    M_B = M_B.reshape(self._Dd, self._d, self._a, self._Dl, self._Du, self._Dr)
    self._gammaB = np.einsum('upardl,r,d,l->paurdl', M_B, self._lambda_l**-1, self._lambda_u**-1, self._lambda_r**-1)


  def update_left(self):
    """
    update lambda_l by applying gate gl to left bond
    only differences from update_right are leg positions
    """
    M_A = np.einsum('paurdl,u,r,d->aurdpl', self._gammaA, self._lambda_u, self._lambda_r, self._lambda_d)
    M_B = np.einsum('paurdl,u,d,l->rpaudl', self._gammaB, self._lambda_d, self._lambda_u, self._lambda_r)
    M_A, self._lambda_l, M_B = svd_SU(M_A, M_B, self._lambda_l, self._gl, self._a*self._Du*self._Dr*self._Dd, self._d, self._Dl)
    M_A = M_A.reshape(self._a, self._Du, self._Dr, self._Dd, self._d, self._Dl)
    self._gammaA = np.einsum('aurdpl,u,r,d->paurdl', M_A, self._lambda_u**-1, self._lambda_r**-1, self._lambda_d**-1)
    M_B = M_B.reshape(self._Dl, self._d, self._a, self._Dd, self._Du, self._Dr)
    self._gammaB = np.einsum('rpaudl,u,d,l->paurdl', M_B, self._lambda_d**-1, self._lambda_u**-1, self._lambda_r**-1)


  def update_up(self):
    """
    update lambda_u by applying gate gu to up bond
    only differences from update_right are leg positions
    """
    M_A = np.einsum('paurdl,r,d,l->ardlpu', self._gammaA, self._lambda_r, self._lambda_d, self._lambda_l)
    M_B = np.einsum('paurdl,u,r,l->dpaurl', self._gammaB, self._lambda_d, self._lambda_l, self._lambda_r)
    M_A, self._lambda_u, M_B = svd_SU(M_A, M_B, self._lambda_u, self._gu, self._a*self._Dr*self._Dd*self._Dl, self._d, self._Du)
    M_A = M_A.reshape(self._a, self._Dr, self._Dd, self._Dl, self._d, self._Du)
    self._gammaA = np.einsum('ardlpu,r,d,l->paurdl', M_A, self._lambda_r**-1, self._lambda_d**-1, self._lambda_l**-1)
    M_B = M_B.reshape(self._Du, self._d, self._a, self._Dd, self._Dl, self._Dr)
    self._gammaB = np.einsum('dpaurl,u,r,l->paurdl', M_B, self._lambda_d**-1, self._lambda_l**-1, self._lambda_r**-1)
