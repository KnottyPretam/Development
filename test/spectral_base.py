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
