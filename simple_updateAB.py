import numpy as np
import scipy.linalg as lg

class SimpleUpdateAB(object):

  def __init__(self, sh, gates, A0=None, B0=None):
    """
    sh: tuple of int, shape of tensor A.
    form (d,a,Du,Dr,Dd,Dl), where a=1 for pure wavefunction and a=d for thermal TN
    shape of B is then (d,a,Dd,Dl,Du,Dr)
    """

    self._d, self._a, self._Du, self._Dr, self._Dd, self._Dl = sh
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


  def update_right(self):
    """
    update lambda_r by applying gate gr to right link
    """
    D_aux = self._a*self._Du*self._Dd*self._Dl    # auxiliary dimension, cut by SVD
    D_eff = self._Dr*self._d                      # effective link dimension after SVD

    # 1) add diagonal lambdas to gammaA
    M_A = np.einsum('paurdl,u,d,l->audlpr', self._gammaA, self._lambda_u, self._lambda_d, self._lambda_l)
    # 2) SVD cut between constant tensors and effective tensor to update
    # hence reduce main SVD to dimension D_eff*d < D_aux*d
    #     \|        \|
    #     -A-    -> -W==M-
    #      |\        |   \
    M_A = M_A.reshape(D_aux, D_eff)
    W_A, sA, M_A = lg.svd(M_A, full_matrices=False)

    # 3) repeat steps 1 and 2 for B
    M_B = np.einsum('paurdl,u,r,d->lpaurd', self._gammaB, self._lambda_d, self._lambda_l, self._lambda_u)
    M_B = B.reshape(D_eff, D_aux)
    M_B, sB, W_B = lg.svd(B, full_matrices=False)

    # 4) construct matrix theta
    #
    #             =MA-lr-MB=
    #                \  /
    #   theta =       gg
    #                /  \
    theta = M_A.reshape(D_eff*self._d, self._Dr)
    theta = np.einsum('ij,j->ij', theta, self._lambda_r)   # add lambda_r, more efficient after SVD
    theta = np.dot(theta, M_B.reshape(self._Dr,self._d*D_eff) )
    theta = theta.reshape(D_eff, self._d, self._d, D_eff).transpose(0,3,1,2).reshape(D_eff**2, self._d**2)
    theta = np.dot(theta, self._gr)
    theta = theta.reshape(D_eff,D_eff,self._d,self._d).swapaxes(1,2).reshape(D_eff*self._d, D_eff*self._d)

    # 5) SVD cut theta and renormalize D_eff*d to D_eff
    # # TODO truncate directly inside SVD
    U,s,V = lg.svd(theta)
    self._lambda_r = s[:self._Dr]    # renormalize lambda_r with SVD weights

    # 6) define new gammaA from W_A and U
    U = np.einsum('i,ij->ij', sA, U[:,:self._Dr].reshape(D_eff, self._d*self._Dr))
    W_A = np.dot(W_A, U).reshape(self._a, self._Du, self._Dd, delf._Dl, self._d, self._Dr)

    # 7) define new gammaA by removing lambdas
    self_gammaA = np.einsum('audldr,u,d,l->paurdl',W_A, self._lambda_u**-1, self._lambda_d**-1, self._lambda_l**-1)

    # 8) repeat steps 6 and 7 for B
    V = np.einsum('ij,j->ij', V[:self._Dr].reshape(self._Dr*self_d, D_eff), sB)
    W_B = np.dot(V, W_B).reshape(self._Dr, self._d, self._a, self._Dd, self._Dl, self_Du)
    self._gammaB = np.einsum('lpaurd,u,r,d->paurdl',W_B, self._lambda_d**-1, self._lambda_l**-1, self._lambda_u**-1)
