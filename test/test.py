#! /usr/bin/env python
import sys,commands,re,os,math
from skimage import data, io, filters
import numpy as np
from img_process_base import imgPTools

imaging = imgPTools()
imArray = imaging.imgSubtract('ref.jpg', '2bars.jpg')
imArray2 = imaging.imgThreshold(imArray, 36, 255)
imgtype = imArray2.dtype
cheksize = imArray.shape

A = np.zeros((4,4))
Bl = [[0, 1], [-2, -3]]
B = np.array(Bl)
A[:,0] = 4
C = A * 3
C[:, 3] = 5
A[:,3] = 2
tA = np.random.rand(4,4)
D, V = np.linalg.eig(B)
minT, maxT, newAr = imaging.imgNorm(tA)
newsum = np.sum(A)

E = A * C
newE = (E - 1) / 2
Esize = E.shape
print('The A array is ' + repr(A)) #print(repr(C)) #print(repr(E)) #print(repr(newE))
#print(' Size of E array is ' + repr(Esize))
#print(' Size of image array is ' + repr(cheksize))
#print('Random Array is : ' + repr(tA))
#print('Min is ' + repr(minT))
#print('Max is ' + repr(maxT))
#print('New normalized array is ' + repr(newAr))
#print('Image Array is ' + repr(imArray))
#print( ' Next Image Array is ' + repr(imArray2))
#print('Image type is ' + repr(imgtype))
#print(repr(imArray))
#print('The Sum of this array is ' + repr(newsum))
print('Eigenvalues ' + repr(D))
Ds = sorted(D, reverse=True)
DsI = np.argsort(D)
print('Sorted Eigenvalues ' + repr(Ds))
print('Eigenvector ' + repr(V))
print('Sorting index for eiganvalues  is ' + repr(DsI))
DD = np.array([ 10, 99, 3, -10, 4, 6 ])
DD2 = DD * -1
DDI = np.argsort(DD2)
Ds2 = sorted(DD, reverse=True)
Ds2I = np.argsort(Ds2)
print('Un sorted array is ' + repr(DD))
print('Indices for sorted array is ' + repr(DDI))
print('Sorted array is ' + repr(Ds2))
print('Sorrted arrays indicies are ' + repr(Ds2I))

DI = np.diag(D)
print('Eigen Identity is ' + repr(DI))
DI2 = np.diag(Ds)
print('Eigen Identity is ' + repr(DI2))
