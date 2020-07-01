import numpy as np
import scipy.linalg as lg

import Env
from ctm_contract import construct_U_half, construct_L_half, construct_D_half, construct_R_half
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
      print(f'initalize CTMRG with chi = {chi}, verbosity = {verbosity} and tiling = {tiling}')
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

  def iterate(self):
    self.up_move()
    self.right_move()
    self.down_move()
    self.left_move()


  def up_move(self):
    if self.verbosity > 0:
      print('\nstart up move')
    # 1) compute isometries for every non-equivalent sites
    # convention : for every move, leg 0 of R and Rt are to be contracted
    for x,y in self._neq_coords:
      R = construct_R_half(self._env.get_T1(x+2,y),  self._env.get_C2(x+3,y),
                           self._env.get_A(x+2,y+1), self._env.get_T2(x+3,y+1),
                           self._env.get_A(x+2,y+2), self._env.get_T2(x+3,y+2),
                           self._env.get_T4(x+2,y+3),self._env.get_C4(x+3,y+3))
      Rt = construct_L_half(self._env.get_C1(x,y),   self._env.get_T1(x+1,y),
                            self._env.get_T4(x,y+1), self._env.get_A(x+1,y+1),
                            self._env.get_T4(x,y+2), self._env.get_A(x+1,y+3),
                            self._env.get_C4(x,y+4),self._env.get_T3(x+1,y+4))
      #        L-0  == 1-R
      #        L         R
      #        L         R  => transpose R
      #        L-1     0-R
      P,Pt = construct_projectors(R.T, Rt, self.chi, self.verbosity)
      self._env.set_projectors(x+1,y,P,Pt)
      del R, Rt

    # 2) renormalize every non-equivalent C1, T1 and C2
    # P != P_list[i] => need all projectors to be constructed at this time
    nC1s,nT1s,nC2s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x,y+1)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC1s[j] = renormalize_C1_up(self._env,x,y,self.verbosity)
      nT1s[j] = renormalize_T1(self._env,x,y,self.verbosity)
      nC2s[j] = renormalize_C2_up(self._env,x,y,self.verbosity)

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
      R = construct_D_half(self._env,x,y,self.verbosity).T
      #      UUUU     0
      #      |  | =>  Rt
      #      1  0     1
      Rt = construct_U_half(self._env,x,y,self.verbosity)
      P,Pt = construct_projectors(R,Rt,self.chi,self.verbosity)
      self._env.set_projectors(x,y+1,P,Pt)
      del R, Rt

    # 2) renormalize tensors by absorbing column
    nC2s,nT2s,nC3s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x-1,y)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC2s[j] = renormalize_C2_right(self._env,x,y,self.verbosity)
      nT2s[j] = renormalize_T2(self._env,x,y,self.verbosity)
      nC3s[j] = renormalize_C3_right(self._env,x,y,self.verbosity)

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
      R = construct_L_half(self._env,x,y,self.verbosity).T
      #      1-R
      #        R
      #        R
      #      0-R
      Rt = construct_R_half(self._env,x,y,self.verbosity)
      P,Pt = construct_projectors(R,Rt,self.chi,self.verbosity)
      self._env.set_projectors(x+1,y+3,P,Pt)
      del R, Rt

    # 2) renormalize every non-equivalent C3, T3 and C4
    nC3s,nT3s,nC4s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x,y-1)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC3s[j] = renormalize_C3_down(self._env,x,y,self.verbosity)
      nT3s[j] = renormalize_T3(self._env,x,y,self.verbosity)
      nC4s[j] = renormalize_C4_down(self._env,x,y,self.verbosity)

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
      R = construct_U_half(self._env,x,y,self.verbosity).T
      #      0  1
      #      |  |
      #      DDDD
      Rt = construct_D_half(self._env,x,y,self.verbosity)
      P,Pt = construct_projectors(R,Rt,self.chi,self.verbosity)
      self._env.set_projectors(x+3,y+1,P,Pt)
      del R, Rt

    # 2) renormalize every non-equivalent C4, T4 and C1
    nC4s,nT4s,nC1s = [None]*self._Nneq, [None]*self._Nneq, [None]*self._Nneq
    if self.verbosity > 0:
      print('Projectors constructed, renormalize tensors')
    for i, (x,y) in enumerate(self._neq_coords):
      j = self._env.get_neq_index(x+1,y)
      if self.verbosity > 1:
        print(f'x = {x}, y = {y}, i = {i}, j = {j}')
      nC4s[j] = renormalize_C4_left(self._env,x,y,self.verbosity)
      nT4s[j] = renormalize_T4(self._env,x,y,self.verbosity)
      nC1s[j] = renormalize_C1_left(self._env,x,y,self.verbosity)

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
