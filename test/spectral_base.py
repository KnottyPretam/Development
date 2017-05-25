#! /usr/bin/env python
import sys,commands,re,os,math
from scipy import signal
import numpy as np

class modalDecompositions(object):
# This class contains modal Decomposition techniques to use 
  def POD(self, U, snapshots):
## This function performs the Proper Orthogonal Decomposition on a Tensor
## This currentlly works for only 3 dimensional matrices
## U is the data matrix 
## snapshots is the cutoff number for total number of sequences calcualted
    dim = U.shape
    nrow = dim[0]
    ncol = dim[1]
    total_modes = dim[2]
    if (total_modes < snapshots):
      total_modes = snapshots

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

    ## This step calculates the spatial POD modes
    podmode = np.zeros((nrow,ncol))
    i = 0
    j = 0
    for i in range (total_modes):
      for j in range (total_modes):
        podmode[:,:,i] = podmode[:, :, i] + sorted_V[j, i] * U[:, :, j]
      mode_factor[i] = 1 / math.sqrt(total_modes * sorted_D[i])
      pom[:, :, i] = podmode[:, :, i] * mode_factor

    ## This step calculates modal energy budget
    E = np.zeros((total_modes,1))
    E[0] = D[0]
    i = 0
    for i in range(1,total_modes):
      E[i] = E[i-1] + D[i]
      Ecomp[i] = np.sum(D) - E[i-1]
    modal_energy = E / np.sum(D)

    ## This step calculates the truncation error
    trun_error = Ecomp / np.sum(D)

    ## This step calculates the temporal coefficients
    a = np.zeros((total_modes, total_modes))
    i = 0
    for i in range(total_modes):
      a[:, i] = ( total_modes * D[i] * V[:,i] ) ** 0.5

    return pom, a, U0, modal_energy, trun_error

  def psd_return(signal, sampling_rate):
## This method uses the built in signal function welch to return relevat parameters
    f, Pxx = signal.welch( signal, sampling_rate)
    peak_locations = np.where(Pxx == np.max(Pxx))
    peak_frequency = f[peak_locations[0] ]
