# coding: utf-8
"""extract time domain features from 1*n EEG signal.
"""

# Authors: RM Peng <rmpeng19@gmail.com>
from pywt import wavedec
import numpy as np
import pandas as pd
import itertools
import math
from scipy.stats import kurtosistest, skew
# import antropy

class timedomain:
   def __init__(self, eegData):
       self.data = np.squeeze(eegData) # 1*n
       #print(self.data.shape)

   def curvelength(self):
       eegdata = self.data
       n_samples = len(eegdata)
       curvelens = 0
       for i in range(n_samples-1):

           temp = eegdata[i+1] - eegdata[i]
           curvelens += temp

       return curvelens

   def nonlinear_energy(self):
       eegdata = self.data
       n_samples = len(eegdata)
       totalenergy = 0
       for i in range(n_samples-2):
           nl = eegdata[i+1]*eegdata[i+1] - eegdata[i]*eegdata[i+2]
           totalenergy +=nl
       nonlinearenery = totalenergy / (n_samples - 2)

       return nonlinearenery

   def rmsAmp(self):
       eegdata = self.data
       n_samples = len(eegdata)
       sum = 0
       for i in range(n_samples):
           sum += eegdata[i]*eegdata[i]
       rmsamp = math.sqrt(sum / n_samples)
       return rmsamp

   def nummaxmin(self):
       cnt = 0
       eegdata = self.data
       n_samples = len(eegdata)
       for i in range(n_samples - 1):
           if math.fabs(eegdata[i] - eegdata[i+1]) < 0.01:
               cnt += 1
       return cnt

   def zerocrossingrate(self,step = 5):
       cnt = 0
       eegdata = self.data.copy()
       eegdata-=eegdata.mean()
       n_samples = len(eegdata)
       for i in range(n_samples - step):

           if eegdata[i] * eegdata[i + step] < 0:
               cnt += 1
       return cnt / n_samples

   def kurtandskew(self):
       kurt,kurt_p = kurtosistest(self.data)
       skewness = skew(self.data)
       # print('skew:',skewness)
       return kurt,skewness

   def hjorthparam(self):

       eeg = self.data
       activity = np.var(eeg)
       # print('activity:',activity)
       sigma = np.std(eeg,ddof=0)

       diff1 = eeg.diff().dropna()  # 1st derivative cross column
       #
       mobility = diff1.std(ddof=0) / sigma
       # print('mob', mobility)
       diff2 = eeg.diff(periods=2).dropna()
       complexity = mobility/(diff2.std(ddof=0) / diff1.std(ddof=0))

       return activity,mobility,complexity

   def emotion(self):
       eegdata = self.data
       n_samples = len(eegdata)

       power = 1/n_samples * sum(eegdata * eegdata.T)
       mean = 1/n_samples * sum(eegdata)
       sigma = eegdata.std(ddof=0)
       d1 = 1/(n_samples -1) * sum(eegdata.diff().abs().dropna())
       nd1 = d1 / sigma
       d2 = 1/(n_samples -1) * sum(eegdata.diff(periods=2).abs().dropna())
       nd2 = d2 / sigma

       return power,mean,d1,  nd1, d2, nd2

   def time_main(self, mysteps = 5):

       cv = self.curvelength()

       nl = self.nonlinear_energy()

       rmsAmp = self.rmsAmp()

       num = self.nummaxmin()

       zcr = self.zerocrossingrate(step = mysteps)

       kurt, skew = self.kurtandskew()

       ac, mb,cp = self.hjorthparam()

       power,mean,d1,nd1,d2,nd2 = self.emotion()

       time_feature =[cv,nl,rmsAmp,num,zcr,kurt,skew,ac, mb,cp,power,mean,d1,nd1,d2,nd2]

       return time_feature
