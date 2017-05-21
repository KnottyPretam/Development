#! /usr/bin/env python
################################################################################
# %OPLICENSE%                                                                  #
#             RESTRICTED PROPRIETARY SOFTWARE AND DATA                         *
#                                                                              *
# This software and the data incorporated herein is licensed, not sold,        *
# and is protected by Federal and State copyright and related intellectual     *
# property laws. It may not be disclosed, copied, reversed engineered or       *
# used in any manner except in strict accordance with the terms and            *
# conditions of a certain License and Sublicense granted by OPINICUS to        *
# ProFlight, and only on the one Citation simulator to which it relates.       *
# Sale, lease or other transfer of the simulator does not authorize the        *
# transferee to use this software and the data incorporated herein             *
# unless strict compliance with the terms of the License and Sublicense        *
# referenced above has been met. A copy of the License and Sublicense is       *
# available from OPINICUS upon request.                                        *
# %OPLICENSE%                                                                  #
################################################################################
import sys,string,commands,operator,re,os,math,gzip,shutil


class textFind(object):

# Finds the location of an unkown string based of the value of a known string
# Example uses for this is to find data points in a text file based off the 
# the variable name within a text file
  def findvalue(self, filename, markerword, linesapart, itemsapart):
  
    i = 0
    markerflag = 0
    fi = open(filename,'r')
    correct_line = []
    markerword_index = []
    for line in fi:
      sline = string.split(line)
      marker_line = string.find(line, markerword)
      if(marker_line != -1):
        correct_line = i + linesapart
#       markerword_index = sline.index(markerword)
#       print('Marker word index is : " +repr(markerword_index))
  
      if(i == correct_line and not markerflag):
        correctindex = itemsapart
        desiredword = sline[correctindex]
        markerflag = 1
  #     print('Correct index is : '+repr(correctindex))
  
      i = i + 1
  
    fi.close()
    return desiredword, correct_line, correctindex

# This function will collect a data array from a text file
  def datagrab(self,filename, rowStart, colStart, stoppulling):
  # If stoppulling is set to 0, datagrab will grab all values until the end of file
    i = 0
    zipcheck = string.find(filename, 'gz')
    pulledarray = []
    returnarray = []
  
    if (zipcheck != -1):
      fi = gzip.open(filename, 'r')
    else:
      fi = open(filename, 'r')
  
    for line in fi:
      sline = string.split(line)
      if (i >= colStart):
        newdata = float(sline[rowStart])
        pulledarray.append(newdata)
      i = i + 1
  
    fi.close()
    if (stoppulling == 0):
      returnarray = pulledarray
    else:
      returnarray = pulledarray[0:stoppulling]
    return returnarray

