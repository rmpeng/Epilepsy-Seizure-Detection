# coding: utf-8
"""extract frequcency domain features from 1*n EEG signal.
"""

# Authors: RM Peng <rmpeng19@gmail.com>
from pywt import wavedec
import numpy as np
import pandas as pd
import itertools
import math
from scipy.stats import kurtosistest, skew
import utils.psd2 as psd

'''
       :param epoch - signal
       :param lvl - frequency levels
       :param nt - length of the signal
       :param fs - sampling frequency
       :param nc - number of channels
'''

class freqdomain:
   def __init__(self, eegData, myfs):
       self.data = np.squeeze(eegData) # 1*n
       self.data = eegData
       self.myfs = myfs #sampling freq

   def psd_features(self):
       data = np.squeeze(self.data)  # 1*n
       return psd.psd2(data, fs = self.myfs)

   def calcNormalizedFFT(self, lvl, nt):
       epoch = self.data
       fs = self.myfs
       """
       Calculates the FFT of the epoch signal.
       Removes the DC component and normalizes the area to 1

       """
       lseg = np.round(nt / fs * lvl).astype('int')
       D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
       D[0, :] = 0  # set the DC component to zero
       D /= D.sum()  # Normalize each channel
       return D

   def calcDSpect(self, lvl, nt, nc):
       epoch = self.data
       fs = self.myfs
       D = self.calcNormalizedFFT(lvl = lvl, nt = nt)
       lseg = np.round(nt / fs * lvl).astype('int')

       dspect = np.zeros((len(lvl) - 1, nc))
       for j in range(len(dspect)):
           dspect[j, :] = 2 * np.sum(D[lseg[j]:lseg[j + 1], :], axis=0)

       return dspect

   def calcSpectralEdgeFreq(self, lvl, nt, nc = 1, percent=0.5):
       # find the spectral edge frequency
       epoch = self.data
       fs = self.myfs
       sfreq = fs
       tfreq = 40
       ppow = percent

       topfreq = int(round(nt / sfreq * tfreq)) + 1
       D = self.calcNormalizedFFT( lvl, nt)
       A = np.cumsum(D[:topfreq, :], axis=1)
       B = A - (A.max() * ppow)
       spedge = np.min(np.abs(B), axis=1)
       spedge = (spedge - 1) / (topfreq - 1) * tfreq

       return spedge

   def SEF(self, percent1,percent2,percent3):
       eeg = self.data
       lvl = np.array([0.4, 4, 8, 12, 30, 70, 180])
       [nc, nt] = eeg.shape

       sef1 = self.calcSpectralEdgeFreq(lvl,nt,nc,percent1)

       sef2 = self.calcSpectralEdgeFreq(lvl,nt,nc,percent2)


       sef3 = self.calcSpectralEdgeFreq(lvl,nt,nc,percent3)


       mean1 = np.mean(sef1)
       mean2 = np.mean(sef2)
       mean3 = np.mean(sef3)
       max1 = np.max(sef1)
       max2 = np.max(sef2)
       max3 = np.max(sef3)
       min1 = np.min(sef1)
       min2 = np.min(sef2)
       min3 = np.min(sef3)
       var1 = np.var(sef1)
       var2 = np.var(sef2)
       var3 = np.var(sef3)

       return mean1,max1,min1,var1,mean2,max2,min2,var2,mean3,max3,min3,var3

   def main_freq(self,percent1= 0.3,percent2=0.5,percent3=0.9):
       mpf, fmax, fmin, fpcntile, Ptotal = self.psd_features()
       mean1, max1, min1, var1, mean2, max2, min2, var2, mean3, max3, min3, var3= self.SEF(percent1=percent1,percent2=percent2,percent3=percent3)
       freq_feature = [mpf, fmax, fmin, Ptotal,mean1, max1, min1, var1, mean2, max2, min2, var2, mean3, max3, min3, var3]
       freq_feature.extend(fpcntile)
       return freq_feature