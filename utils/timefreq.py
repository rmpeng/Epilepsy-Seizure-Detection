# coding: utf-8
"""extract time-frequency features from 1*n EEG signal
"""

# Authors: RM Peng <rmpeng19@gmail.com>
import numpy as np
import pandas as pd
import pywt
import segyio as segyio
from scipy.stats import kurtosis
import statistics
import scipy.signal
from scipy.signal import hilbert, butter, filtfilt,savgol_filter
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
from scipy import interpolate
import argparse
#from pyhht.emd import *
from pyhht.visualization import plot_imfs
from pylab import rcParams
import pyhht

def emd_trace(eeg,nimfs):
    """Trace EMD : Emperical Mode Decomposition"""
    eeg = np.squeeze(eeg.T)

    #print(eeg.shape)


    decomposer = pyhht.emd.EMD(eeg,n_imfs=nimfs)
    imfs = decomposer.decompose()
    # plot_imfs(tr100[0,:], imfs, ioriginal)
    #envmoy, nem, nzm, amp = decomposer.mean_and_amplitude(eeg)
    # mean_amp[0] is mean amplitude
    # amp[3] is mode or envelop
    return imfs#,envmoy, nem, nzm, amp


def smooth(a, wlen) :
    if wlen % 2 == 0:  #window has to be odd
        wlen +=1
    asmth = savgol_filter(a, wlen, 3) # window size 51, polynomial order 3
    return asmth


def inst_attributes(tr,freq_sample,smoothwlen):
    """Compute instantaneous attributes"""
    analytic_signal = hilbert(tr)
    aenv = np.abs(analytic_signal)
    iph = np.unwrap(np.angle(analytic_signal))
    ifreq = np.abs((np.diff(iph,prepend=0) / (2.0*np.pi) * freq_sample))
    ifreqsmth = smooth(ifreq,smoothwlen)

    return aenv,iph,ifreqsmth

def comp_moment(feature):
    '''this function computes the moments like mean, standard deviation
    and kutosis of the obtained feature vector'''
    step = int(len(feature)/2)
    # variables to be used inside loops
    avg_temp = np.zeros([2])
    stn_dev_temp = np.zeros([2])
    kurto_temp = np.zeros([2])

    for i in range(int(len(feature)/step)):
        avg_temp[i] = np.mean(feature[step*i:step*(i+1)])
        stn_dev_temp[i] = statistics.stdev(feature[step*i:step*(i+1)])
        kurto_temp[i] = kurtosis(feature[step*i:step*(i+1)])
    return avg_temp, stn_dev_temp, kurto_temp

class timefreq:
    def __init__(self, eegData, myfs):
        self.data = np.squeeze(eegData)
        self.fs = myfs
        self.sample = len(eegData)

    def district_wavelet(self):
        #cA, cD = pywt.dwt(self.data, 'db4')
        [a, d1, d2, d3, d4] = pywt.wavedec(self.data, 'haar', level=4)

        #approximation coefficient
        avg_temp, stn_dev_temp, kurto_temp = comp_moment(a)
        avg_a = np.mean(avg_temp)
        stn_dev_a = np.mean(stn_dev_temp)
        kurto_a = np.mean(kurto_temp)

        #d1 coffiecient
        avg_temp, stn_dev_temp, kurto_temp = comp_moment(d1)
        avg_d1 = np.mean(avg_temp)
        stn_dev_d1  = np.mean(stn_dev_temp)
        kurto_d1  = np.mean(kurto_temp)

        # d2 coffiecient
        avg_temp, stn_dev_temp, kurto_temp = comp_moment(d2)
        avg_d2 = np.mean(avg_temp)
        stn_dev_d2 = np.mean(stn_dev_temp)
        kurto_d2 = np.mean(kurto_temp)

        # d3 coffiecient
        avg_temp, stn_dev_temp, kurto_temp = comp_moment(d3)
        avg_d3 = np.mean(avg_temp)
        stn_dev_d3 = np.mean(stn_dev_temp)
        kurto_d3 = np.mean(kurto_temp)

        # d4 coffiecient
        avg_temp, stn_dev_temp, kurto_temp = comp_moment(d4)
        avg_d4 = np.mean(avg_temp)
        stn_dev_d4 = np.mean(stn_dev_temp)
        kurto_d4 = np.mean(kurto_temp)

        return avg_a,avg_d1,avg_d2,avg_d3,avg_d4,\
               stn_dev_a,stn_dev_d1,stn_dev_d2,stn_dev_d3,stn_dev_d4,\
               kurto_a,kurto_d1,kurto_d2,kurto_d3,kurto_d4

    def emd_hht(self, smoothlength, nimfs):
        tr = self.data
        fs = self.fs
        imf_out = emd_trace(tr,nimfs)

        trenv,triph,trifreq = inst_attributes(tr,freq_sample=fs,smoothwlen=smoothlength)
        trenv_avg = np.mean(trenv)
        triph_avg = np.mean(triph)
        trifreq_avg = np.mean(trifreq)
        trenv_var = np.var(trenv)
        triph_var = np.var(triph)
        trifreq_var = np.var(trifreq)
        imf0 = imf_out[0, :].T
        imf0env, imf0iph, imf0ifreq = inst_attributes(imf0, freq_sample=fs, smoothwlen=smoothlength)
        imf0env_avg = np.mean(imf0env)
        imf0iph_avg = np.mean(imf0iph)
        imf0ifreq_avg = np.mean(imf0ifreq)
        imf0env_var = np.var(imf0env)
        imf0iph_var = np.var(imf0iph)
        imf0ifreq_var = np.var(imf0ifreq)

        imf1 = imf_out[1, :].T
        imf1env, imf1iph, imf1ifreq = inst_attributes(imf1, freq_sample=fs, smoothwlen=smoothlength)
        imf1env_avg = np.mean(imf1env)
        imf1iph_avg = np.mean(imf1iph)
        imf1ifreq_avg = np.mean(imf1ifreq)
        imf1env_var = np.var(imf1env)
        imf1iph_var = np.var(imf1iph)
        imf1ifreq_var = np.var(imf1ifreq)

        imf2 = imf_out[2, :].T
        imf2env, imf2iph, imf2ifreq = inst_attributes(imf2, freq_sample=fs, smoothwlen=smoothlength)
        imf2env_avg = np.mean(imf2env)
        imf2iph_avg = np.mean(imf2iph)
        imf2ifreq_avg = np.mean(imf2ifreq)
        imf2env_var = np.var(imf2env)
        imf2iph_var = np.var(imf2iph)
        imf2ifreq_var = np.var(imf2ifreq)

        imf3 = imf_out[3, :].T
        imf3env, imf3iph, imf3ifreq = inst_attributes(imf3, freq_sample=fs, smoothwlen=smoothlength)
        imf3env_avg = np.mean(imf3env)
        imf3iph_avg = np.mean(imf3iph)
        imf3ifreq_avg = np.mean(imf3ifreq)
        imf3env_var = np.var(imf3env)
        imf3iph_var = np.var(imf3iph)
        imf3ifreq_var = np.var(imf3ifreq)

        return trenv_avg ,triph_avg,trifreq_avg,trenv_var,\
        triph_var,trifreq_var,imf0env_avg , imf0iph_avg , imf0ifreq_avg,\
        imf0env_var,imf0iph_var, imf0ifreq_var,imf1env_avg , imf1iph_avg , imf1ifreq_avg,\
        imf1env_var,imf1iph_var, imf1ifreq_var,imf2env_avg , imf2iph_avg , imf2ifreq_avg,\
        imf2env_var,imf2iph_var, imf2ifreq_var,imf3env_avg , imf3iph_avg , imf3ifreq_avg,\
        imf3env_var,imf3iph_var, imf3ifreq_var


    def main_tf(self,smoothwindow,nimfs=4):
        avg_a, avg_d1, avg_d2, avg_d3, avg_d4,\
        stn_dev_a, stn_dev_d1, stn_dev_d2, stn_dev_d3, stn_dev_d4, \
        kurto_a, kurto_d1, kurto_d2, kurto_d3, kurto_d4 =self.district_wavelet()

        trenv_avg, triph_avg, trifreq_avg, trenv_var, \
        triph_var, trifreq_var, imf0env_avg, imf0iph_avg, imf0ifreq_avg, \
        imf0env_var, imf0iph_var, imf0ifreq_var, imf1env_avg, imf1iph_avg, imf1ifreq_avg, \
        imf1env_var, imf1iph_var, imf1ifreq_var, imf2env_avg, imf2iph_avg, imf2ifreq_avg, \
        imf2env_var, imf2iph_var, imf2ifreq_var, imf3env_avg, imf3iph_avg, imf3ifreq_avg, \
        imf3env_var, imf3iph_var, imf3ifreq_var= self.emd_hht(smoothlength=smoothwindow,nimfs= nimfs)

        tf_feature = [avg_a, avg_d1, avg_d2, avg_d3, avg_d4,
                      stn_dev_a, stn_dev_d1, stn_dev_d2, stn_dev_d3, stn_dev_d4,
                      kurto_a, kurto_d1, kurto_d2, kurto_d3, kurto_d4,
                      trenv_avg, triph_avg, trifreq_avg, trenv_var,triph_var, trifreq_var,
                      imf0env_avg, imf0iph_avg, imf0ifreq_avg,imf0env_var, imf0iph_var, imf0ifreq_var,
                      imf1env_avg, imf1iph_avg, imf1ifreq_avg,imf1env_var, imf1iph_var, imf1ifreq_var,
                      imf2env_avg, imf2iph_avg, imf2ifreq_avg,imf2env_var, imf2iph_var, imf2ifreq_var,
                      imf3env_avg, imf3iph_avg, imf3ifreq_avg,imf3env_var, imf3iph_var, imf3ifreq_var]

        return tf_feature








