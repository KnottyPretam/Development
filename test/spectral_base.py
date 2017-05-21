#! /usr/bin/env python
import sys,commands,re,os,math
from skimage import data, io, filters
import numpy as np

class modalDecompositions(object):
# This class contains modal Decomposition techniques to use 
  def POD(self, U, instances):
## This function performs the Proper Orthogonal Decomposition on a Tensor
## This currentlly works for only 3 dimensional matrices
    dim = U.shape
    nrow = dim[0]
    ncol = dim[1]
    total_modes = dim[2]

    ## This step calcluates the base mode, U0  
    Usum = np.zeros((nrow, ncol))
    k = 0
    for k in range(total_modes):
      Usum = Usum + U[:, :, k]

    U0 = Usum / total_modes

    ## This step calculates the Correlation matrix, C
    i = 0
    j = 0
    C = np.zeros((total_modes, total_modes))
    for i in range(total_modes):
      for j in range(total_modes):
        C[i, j] = np.sum( U[:, :, i] * U[:, :, j])
        C[j, i] = C[i, j]
    C_norm = C / (total_modes)

    ## This step calculates the eigenvalues D, and eigen vector V
    D, V = np.linalg.eig(C_norm)
    DD = D * -1
    sorted_index = np.argsort(DD)
    sorted_V = V[:, :, sorted_index]
    sorted_D = sorted(D, reverse=True)

    ## This step calculated the spatial POD modes
    podmode = np.zeros((nrow,ncol))
    i = 0
    j = 0
    for i in range (total_modes):
      for j in range (total_modes):
        podmode[:,:,i] = podmode[:, :, i] + sorted_V[j, i] * U[:, :, j]
      mode_factor[i] = 1 / math.sqrt(total_modes * sorted_D[i])
      pom[:, :, i] = podmode[:, :, i] * mode_factor
