import numpy as np
import scipy.linalg as lg

import Env
from ctm_contract import contract_U_half, contract_L_half, contract_D_half, contract_R_half
from ctm_renormalize import *
import rdm

class CTMRG(object):
  # convention: legs and tensors are taken clockwise.
  #    C1-T1-T1-C2
  #    |  |  |   |
  #    T4-a--a--T2
  #    |  |  |   |
  #    T4-a--a--T2
  #    |  |  |   |
  #    C4-T3-T3-C3

  def __init__(self,tensors,tiling,chi,verbosity=0):
    self.verbosity = verbosity
    if self.verbosity > 0:
      print(f'initalize CTMRG with chi = {chi}, self.verbosity = {self.verbosity} and tiling = {tiling}')
    self.chi = chi
    self._env = Env.Env(tensors,tiling,chi)
    self._neq_coords = self._env.neq_coords
    self._Nneq = len(self._neq_coords)
    if self.verbosity > 0:
      print('CTMRG constructed')

  @property
  def Lx(self):
    return self._env.Lx

  @property
  def Ly(self):
    return self._env.Ly

  @property
  def env(self):
    return self._env


  def print_tensor_shapes(self):
    print("C1 T1 C2 // T4 A T2 // C4 T3 C4")
    for (x,y) in self._neq_coords:
      print(f'({x},{y}):', self._env.get_C1(x,y).shape,
      self._env.get_T1(x+1,y).shape,    self._env.get_C2(x+2,y).shape,
      self._env.get_T4(x,y+1).shape,    self._env.get_A(x+1,y+1).shape,
      self._env.get_T2(x+2,y+1).shape,  self._env.get_C4(x,y+3).shape,
      self._env.get_T3(x+1,y+2).shape,  self._env.get_C3(x+2,y+2).shape)


  def iterate(self):
    self.up_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()
    self.right_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()
    self.down_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()
    self.left_move()
    if self.verbosity > 1:
      self.print_tensor_shapes()


  def up_move(self):
    if self.verbosity > 0:
      print('\nstart up move')
    # 1) compute isometries for every non-equivalent sites
    # convention : for every move, leg 0 of R and Rt are to be contracted
    for x,y in self._neq_coords:
      R = contract_R_half(self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                           self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                           self._env.get_T3(x+2,y+3),self._env.get_C3(x+3,y+3))
      Rt = contract_L_half(self._env.get_C1(x,y),   self._env.get_T1(x+1,y),
                            self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                            self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                            self._env.get_C4(x,y+3),self._env.get_T3(x+1,y+3))
      #        L-0  == 1-R
      #        L         R
      #        L         R  => transpose R
      #        L-1     0-R
      P,Pt = construct_projectors(R.T, Rt, self.chi)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.set_projectors(x,y,P,Pt)
      del R, Rt

    # 2) renormalize every non-equivalent C1, T1 and C2
    # P != P_list[i] => need all projectors to be constructed at this time
    nC1s,nT1s,nC2s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x,y+1)
      nC1s[j] = renormalize_C1_up(self._env.get_C1(x,y),
                                self._env.get_T4(x,y+1),self._env.get_P(x+1,y))

      nT1s[j] = renormalize_T1(self._env.get_Pt(x,y), self._env.get_T1(x,y),
                                self._env.get_A(x,y+1), self._env.get_P(x+1,y))

      nC2s[j] = renormalize_C2_up(self._env.get_C2(x,y),
                                self._env.get_T2(x,y+1), self._env.get_Pt(x,y))
      if self.verbosity > 1:
        print(f'(x,y) = ({x},{y}) i = {i}, j = {j}, C1.shape = {nC1s[j].shape}, T1.shape = {nT1s[j].shape}, C2.shape = {nC2s[j].shape}')

    # 3) store renormalized tensors in the environment
    # renormalization reads C1[x,y] but writes C1[x,y+1]
    # => need to compute every renormalized tensors before storing any of them
    self._env.reset_projectors()
    self._env.neq_C1s = nC1s
    self._env.neq_T1s = nT1s
    self._env.neq_C2s = nC2s
    if self.verbosity > 0:
      print('up move completed')


  def right_move(self):
    if self.verbosity > 0:
      print('\nstart right move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #      0  1    0
      #      |  | => R
      #      DDDD    1
      R = contract_D_half(self._env.get_T4(x,y+2),   self._env.get_A(x+1,y+2),
                          self._env.get_A(x+2,y+2),  self._env.get_T2(x+3,y+2),
                          self._env.get_C4(x,y+3),   self._env.get_T3(x+1,y+3),
                          self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))
      #      UUUU     0
      #      |  | =>  Rt
      #      1  0     1
      Rt = contract_U_half(self._env.get_C1(x,y),    self._env.get_T1(x+1,y),
                           self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_T4(x,y+1),  self._env.get_A(x+1,y+1),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1))
      P,Pt = construct_projectors(R.T,Rt,self.chi)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.set_projectors(x,y+1,P,Pt)
      del R, Rt

    # 2) renormalize tensors by absorbing column
    nC2s,nT2s,nC3s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x-1,y)
      nC2s[j] = renormalize_C2_right(self._env.get_C2(x,y),
                               self._env.get_T1(x-1,y), self._env.get_P(x,y+1))

      nT2s[j] = renormalize_T2(self._env.get_Pt(x,y), self._env.get_A(x-1,y),
                                 self._env.get_T2(x,y), self._env.get_P(x,y+1))

      nC3s[j] = renormalize_C3_right(self._env.get_C3(x,y),
                                self._env.get_T3(x-1,y), self._env.get_Pt(x,y))
      if self.verbosity > 1:
        print(f'(x,y) = ({x},{y}) i = {i}, j = {j}, C2.shape = {nC2s[j].shape}, T2.shape = {nT2s[j].shape}, C3.shape = {nC3s[j].shape}')

    # 3) store renormalized tensors in the environment
    self._env.reset_projectors()
    self._env.neq_C2s = nC2s
    self._env.neq_T2s = nT2s
    self._env.neq_C3s = nC3s
    if self.verbosity > 0:
      print('right move completed')


  def down_move(self):
    if self.verbosity > 0:
      print('\nstart down move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #        L-0      L-1
      #        L        L
      #        L    =>  L
      #        L-1      L-0
      R = contract_L_half(self._env.get_C1(x,y),     self._env.get_T1(x+1,y),
                            self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                            self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                            self._env.get_C4(x,y+3), self._env.get_T3(x+1,y+3))
      #      1-R
      #        R
      #        R
      #      0-R
      Rt = contract_R_half(self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                           self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                           self._env.get_T3(x+2,y+3),self._env.get_C3(x+3,y+3))

      P,Pt = construct_projectors(R.T,Rt,self.chi)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.set_projectors(x+1,y+3,P,Pt)
      del R, Rt

    # 2) renormalize every non-equivalent C3, T3 and C4
    nC3s,nT3s,nC4s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x,y-1)
      nC3s[j] = renormalize_C3_down(self._env.get_C3(x,y),
                               self._env.get_T2(x,y-1), self._env.get_P(x-1,y))

      nT3s[j] = renormalize_T3(self._env.get_Pt(x,y), self._env.get_T3(x,y),
                                 self._env.get_A(x,y-1), self._env.get_P(x-1,y))

      nC4s[j] = renormalize_C4_down(self._env.get_C4(x,y),
                                self._env.get_T4(x,y-1), self._env.get_Pt(x,y))
      if self.verbosity > 1:
        print(f'(x,y) = ({x},{y}) i = {i}, j = {j}, C3.shape = {nC3s[j].shape}, T3.shape = {nT3s[j].shape}, C4.shape = {nC4s[j].shape}')

    # 3) store renormalized tensors in the environment
    self._env.reset_projectors()
    self._env.reset_projectors()
    self._env.neq_C3s = nC3s
    self._env.neq_T3s = nT3s
    self._env.neq_C4s = nC4s
    if self.verbosity > 0:
      print('down move completed')


  def left_move(self):
    if self.verbosity > 0:
      print('\nstart left move')
    # 1) compute isometries for every non-equivalent sites
    for x,y in self._neq_coords:
      #      UUUU      1
      #      |  |  =>  R
      #      1  0      0
      R = contract_U_half(self._env.get_C1(x,y),    self._env.get_T1(x+1,y),
                           self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_T4(x,y+1),  self._env.get_A(x+1,y+1),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1))
      #      0  1
      #      |  |
      #      DDDD
      Rt = contract_D_half(self._env.get_T4(x,y+2),   self._env.get_A(x+1,y+2),
                          self._env.get_A(x+2,y+2),  self._env.get_T2(x+3,y+2),
                          self._env.get_C4(x,y+3),   self._env.get_T3(x+1,y+3),
                          self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))
      P,Pt = construct_projectors(R.T,Rt,self.chi)
      if self.verbosity > 1:
        print(f'constructed projectors: R.shape = {R.shape}, Rt.shape = {Rt.shape}, P.shape = {P.shape}, Pt.shape = {Pt.shape}')
      self._env.set_projectors(x,y+1,P,Pt)
      del R, Rt

    # 2) renormalize every non-equivalent C4, T4 and C1
    nC4s,nT4s,nC1s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x+1,y)
      nC4s[j] = renormalize_C4_left(self._env.get_C4(x,y),
                               self._env.get_T3(x+1,y), self._env.get_P(x,y-1))

      nT4s[j] = renormalize_T4(self._env.get_Pt(x,y), self._env.get_T4(x,y),
                                self._env.get_A(x+1,y), self._env.get_P(x,y-1))

      nC1s[j] = renormalize_C1_left(self._env.get_C1(x,y),
                                self._env.get_T1(x+1,y), self._env.get_Pt(x,y))
      if self.verbosity > 1:
        print(f'(x,y) = ({x},{y}) i = {i}, j = {j}, C4.shape = {nC4s[j].shape}, T4.shape = {nT4s[j].shape}, C1.shape = {nC1s[j].shape}')

    # 3) store renormalized tensors in the environment
    self._env.reset_projectors()
    self._env.neq_C4s = nC4s
    self._env.neq_T4s = nT4s
    self._env.neq_C1s = nC1s

    if self.verbosity > 0:
      print('left move completed')


  def compute_rdm1x1(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 1x1 with C1 coord = ({x},{y})')
    return rdm.rdm_1x1(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                         self._env.get_C2(x+2,y), self._env.get_T4(x,y+1),
                         self._env.get_A(x+1,y+1), self._env.get_T2(x+2,y+1),
                         self._env.get_C4(x,y+2), self._env.get_T3(x+1,y+2),
                         self._env.get_C3(x+2,y+2))


  def compute_rdm1x2(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 1x2 with C1 coord = ({x},{y})')
    return rdm.rdm_1x2(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                          self._env.get_T1(x+2,y), self._env.get_C2(x+3,y),
                          self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                          self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                          self._env.get_C4(x,y+2), self._env.get_T3(x+1,y+2),
                          self._env.get_T3(x+2,y+2), self._env.get_C3(x+3,y+2))



  def compute_rdm2x1(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 2x1 with C1 coord = ({x},{y})')
    return rdm.rdm_2x1(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                          self._env.get_C2(x+2,y), self._env.get_T4(x,y+1),
                          self._env.get_A(x+1,y+1), self._env.get_T2(x+2,y+1),
                          self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                          self._env.get_T2(x+2,y+2), self._env.get_C4(x,y+3),
                          self._env.get_T3(x+1,y+3), self._env.get_C3(x+2,y+3))


  def compute_rdm2x2(self,x=0,y=0):
    if self.verbosity > 0:
      print(f'Compute rdm 2x2 with C1 coord = ({x},{y})')
    return rdm.rdm_2x2(self._env.get_C1(x,y), self._env.get_T1(x+1,y),
                       self._env.get_T1(x+2,y), self._env.get_C2(x+3,y),
                       self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                       self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                       self._env.get_T4(x,y+2), self._env.get_A(x+1,y+2),
                       self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                       self._env.get_C4(x,y+3), self._env.get_T3(x+1,y+3),
                       self._env.get_T3(x+2,y+3), self._env.get_C3(x+3,y+3))
