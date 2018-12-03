#!/home/liulab/shin/python/bin/python
# 
# WDen.py
#
# WDen.py contains functions for wavelet denoising The functions are named following 
# the naming conventions used by MATLAB Wavelet Toolbox.
#
# Programmed by Hyunjin Shin 
#

import numpy as np
import math
import pywt
import sys
      
#################################
#
# wden(x, tptr, sorh, scal, n, wname) does wavelet denoising. 
#
# x, input signal to be denoised
# tptr, threshold selection rule. See thselect.
# sorh, threshold type. See wthresh
# scal = 'one', for no threshold rescaling
#      = 'sln', for rescaling using a single estimation of level noise based 
#               on the first detail coefficients
#      = 'mln', for rescaling done using level dependent estimation
# wname, wavelet name
#
def wden(x, tptr, sorh, scal, n, wname):
    
    # epsilon stands for a very small number
    eps = 2.220446049250313e-16
    
    # decompose the input signal. Symetric padding is given as a default.
    coeffs = pywt.wavedec(x, wname,'sym', n)
    
    # threshold rescaling coefficients
    if scal == 'one':
        s = 1
    elif scal == 'sln':
        s = np.squeeze(np.ones((1,n))) * wnoisest(coeffs)
    elif scal == 'mln':
        s = wnoisest(coeffs, level = n)
    else: 
        raise ValueError('Invalid value for scale, scal = %s' %(scal))
    
    # wavelet coefficients thresholding
    coeffsd = [coeffs[0]]
    for i in range(0, n):
        if tptr == 'sqtwolog' or tptr == 'minimaxi':
            th = thselect(coeffs, tptr)
        else:
            if len(s) == 1:
                if s < math.sqrt(eps) * max(coeffs[1+i]): th = 0
                else: th = thselect(coeffs[1+i]/s, tptr)
            else:
                if s[i] < math.sqrt(eps) * max(coeffs[1+i]): th = 0
                else: th = thselect(coeffs[1+i]/s[i], tptr)

        # rescaling
        th = th * s[i]
        
        coeffsd.append(np.array(wthresh(coeffs[1+i], sorh, th)))
    # wavelet reconstruction 
    xdtemp = pywt.waverec(coeffsd, wname, 'sym')
    
    # get rid of the extended part for wavelet decomposition
    extlen = int(abs(len(x)-len(xdtemp))/2)
    xd = xdtemp[extlen:len(x)+extlen]
    
    return xd
    
    
#################################
#
# thselect(x, tptr) returns threshold x-adapted value using selection rule defined by string tptr.
# 
# tptr = 'rigrsure', adaptive threshold selection using principle of Stein's Unbiased Risk Estimate.
#        'heursure', heuristic variant of the first option.
#        'sqtwolog', threshold is sqrt(2*log(length(X))).
#        'minimaxi', minimax thresholding. 
        
def thselect(c, tptr):
    x = np.zeros((0,0))
    for i in range(len(c)):
        x = np.append(x,c[i])
    l = len(x)
    
    if tptr == 'rigrsure':
        sx2 = [sx*sx for sx in math.absolute(x)]
        sx2.sort()
        cumsumsx2 = math.cumsum(sx2)
        risks = []
        for i in range(0, l):
            risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
        mini = math.argmin(risks)
        th = math.sqrt(sx2[mini])
    if tptr == 'heursure':
        hth = math.sqrt(2*math.log(l))
        
        # get the norm of x
        normsqr = np.dot(x, x)
        eta = 1.0*(normsqr-l)/l
        crit = (math.log(l,2)**1.5)/math.sqrt(l)
        
        ### DEBUG
#        print "crit:", crit
#        print "eta:", eta
#        print "hth:", hth
        ###
        
        if eta < crit: th = hth
        else: 
            sx2 = [sx*sx for sx in math.absolute(x)]
            sx2.sort()
            cumsumsx2 = math.cumsum(sx2)
            risks = []
            for i in range(0, l):
                risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
            mini = math.argmin(risks)
            
            
            ### DEBUG
#            print "risk:", risks[mini]
#            print "best:", mini
#            print "risks[222]:", risks[222]
            ###
            
            rth = math.sqrt(sx2[mini])
            th = min(hth, rth)     
    elif tptr == 'sqtwolog':
        th = math.sqrt(2*math.log(l))
    elif tptr == 'minimaxi':
        if l < 32:
            th = 0
        else:
            th = 0.3936 + 0.1829*math.log(l, 2)
    else:
        raise ValueError('Invalid value for threshold selection rule, tptr = %s' %(tptr))
    
    return th
    
#################################
#
# wthresh(x, sorh, t) returns the soft (sorh = 'soft') or hard (sorh = 'hard') 
# thresholding of x, the given input vector. t is the threshold.
# sorh = 'hard', hard trehsholding
# sorh = 'soft', soft thresholding
#    
    
def wthresh(x, sorh, t):
    
    if sorh == 'hard':
        y = [e*(abs(e) >= t) for e in x]
    elif sorh == 'soft':
        y = [((e<0)*-1.0 + (e>0))*((abs(e)-t)*(abs(e) >= t)) for e in x]
    else:
        raise ValueError('Invalid value for thresholding type, sorh = %s' %(sorh))
    
    return y

#################################
#
# wnoisest(coeffs, level = None) estimates the variance(s) of the given detail(s)
#
# coeffs = [CAn, CDn, CDn-1, ..., CD1], multi-level wavelet coefficients
# level, decomposition level. None is the default.
#

def wnoisest(coeffs, level= None):
    
    l = len(coeffs) - 1
    stdc = np.zeros((0,0))
    if level == None:
        sig = [abs(s) for s in coeffs[-1]]
        stdc = np.append(stdc,np.median(sig)/0.6745)
    else:
        for i in range(0, l):
            sig = [abs(s) for s in coeffs[1+i]]
            stdc = np.append(stdc, np.median(sig) / 0.6745)
    
    return stdc

#################################
#
# median(data) returns the median of data
#
# data, a list of numbers.
#       
 
# def median(data):
#
#     temp = data[:]
#     temp.sort()
#     dataLen = len(data)
#     if dataLen % 2 == 0: # even number of data points
#         med = (temp[int(dataLen/2) -1] + temp[int(dataLen/2)])/2.0
#     else:
#         med = temp[int(dataLen/2)]
#
#     return med
    