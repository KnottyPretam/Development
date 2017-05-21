#! /usr/bin/env python
import sys,commands,re,os,math
from skimage import data, io, filters
import numpy as np

class imgPTools(object):
# This class contains addition image processing functions /tools which are not
# not part of python's native image processing toolkit 
  def imgSubtract(self, bg_image, test_image ):
    original = io.imread(test_image)
    background = io.imread(bg_image)
#   cleanimage = original - background
    cleanimage = background - original
    io.imsave('newimage.jpg',cleanimage)
    return cleanimage

  def imgThreshold(self, test_image, lowerBound, upperBound):
    cleanimage = (test_image >= lowerBound) & (test_image <= upperBound)
    convertedimage = 255 * cleanimage.view('uint8')
    io.imsave('newThreshholdimage.jpg',convertedimage)
    return convertedimage

  def imgNorm(self, image_array):
    minvalue = np.nanmin(image_array)
    maxvalue = np.nanmax(image_array)
    norm_image = (image_array - minvalue) / (maxvalue - minvalue)
    return minvalue, maxvalue, norm_image
