# coding: utf-8
"""This program aims to extract features of time domain, frequency domain, time-frequency domain and nonlinear analysis.4
After extracting features from different domains, the features will saved as dictionary in one file for each state.
"""

# Authors: RM Peng <rmpeng19@gmail.com>

import matplotlib.pyplot as plt
import os,sys
import csv
from utils import timedomain,freqdomain,timefreq,nonlinear
import pandas as pd
import math
import numpy as np
import scipy.io as scio

##load data and give labels
folder1 = 'dataset\Bonn\A_Z'
folder2 = 'dataset\Bonn\B_O'
folder3 = 'dataset\Bonn\C_N'
folder4 = 'dataset\Bonn\D_F'
folder5 = 'dataset\Bonn\E_S'
datasetA = []
for file in os.listdir(folder1):
    f1 = folder1+ '\\' + file
    datasetA.append(f1)
datasetA = sorted(datasetA)

datasetB = []
for file in os.listdir(folder2):
    f2 = folder2+ '\\' + file
    datasetB.append(f2)
datasetB = sorted(datasetB)

datasetC = []
for file in os.listdir(folder3):
    f3 = folder3+ '\\' + file
    datasetC.append(f3)
datasetC = sorted(datasetC)

datasetD = []
for file in os.listdir(folder4):
    f4 = folder4+ '\\' + file
    datasetD.append(f4)
datasetD = sorted(datasetD)

datasetE = []
for file in os.listdir(folder5):
    f5 = folder5 + '\\'+file
    datasetE.append(f5)
datasetE = sorted(datasetE)

normal = []
for i in range(len(datasetA)):
    x = pd.read_table(datasetA[i],header = None)
    normal.append(x)

for i in range(len(datasetB)):
    x = pd.read_table(datasetB[i],header = None)
    normal.append(x)

inter = []
for i in range(len(datasetC)):
    x = pd.read_table(datasetC[i],header = None)
    inter.append(x)

for i in range(len(datasetD)):
    x = pd.read_table(datasetD[i],header = None)
    inter.append(x)

ictal = []
for i in range(len(datasetE)):
    x = pd.read_table(datasetE[i],header = None)
    ictal.append(x)

##param
fs = 173 #sampling rate
Tau = 4
M = 10
R = 0.3
Band = np.arange(1,86)
DE = 10

#extract features for normal states
time_feature = []
freq_feature = []
tf_feature = []
nonlinear_feature = []

for j in range(len(normal)):

    trans = normal[j].T

    temp1 = timedomain.timedomain(trans)
    matrix1 = np.array(temp1.time_main(mysteps=5))
    time_feature.append(matrix1)# time domain features

    temp2 = freqdomain.freqdomain(trans,myfs=fs)
    matrix2 = np.array(temp2.main_freq(percent1=0.5,percent2=0.8,percent3=0.95))
    freq_feature.append(matrix2)#frequency domain features

    temp3 = timefreq.timefreq(trans,myfs=fs)
    matrix3 = np.array(temp3.main_tf(smoothwindow=100))
    tf_feature.append(matrix3)# time-frequency domain features

    temp4 = nonlinear.nonlinear(trans, myfs=fs)
    matrix4 = np.array(temp4.nonlinear_main(tau=Tau,m=M,r=R,de =DE,n_perm=4,n_lya=40,band=Band))
    nonlinear_feature.append(matrix4)# nonlinear analysis

time_feature = np.array(time_feature)
freq_feature = np.array(freq_feature)
tf_feature = np.array(tf_feature)
nonlinear_feature = np.array(nonlinear_feature)

file1= 'dataset\\bonn_feature\\normal_feature.npz'
np.savez(file1,time=time_feature,freq=freq_feature,tf = tf_feature,entropy = nonlinear_feature)
print('normal features saved!')

#extract features for inter-ictal states
time_feature = []
freq_feature = []
tf_feature = []
nonlinear_feature = []
for j in range(len(inter)):
    trans = inter[j].T
    temp1 = timedomain.timedomain(trans)
    matrix1 = np.array(temp1.time_main(mysteps=5))
    time_feature.append(matrix1)

    temp2 = freqdomain.freqdomain(trans, myfs=fs)
    matrix2 = np.array(temp2.main_freq(percent1=0.5, percent2=0.8, percent3=0.95))
    freq_feature.append(matrix2)

    temp3 = timefreq.timefreq(trans, myfs=fs)
    matrix3 = np.array(temp3.main_tf(smoothwindow=100))
    tf_feature.append(matrix3)

    temp4 = nonlinear.nonlinear(trans, myfs=fs)
    matrix4 = np.array(temp4.nonlinear_main(tau=Tau, m=M, r=R, de=DE, n_perm=4, n_lya=4, band=Band))
    nonlinear_feature.append(matrix4)

time_feature = np.array(time_feature)
freq_feature = np.array(freq_feature)
tf_feature = np.array(tf_feature)
nonlinear_feature = np.array(nonlinear_feature)

file2= 'dataset\\bonn_feature\\inter_feature.npz'
np.savez(file2,time=time_feature,freq=freq_feature,tf = tf_feature,entropy = nonlinear_feature)
print('inter features saved!')

#extract features for ictal states
time_feature = []
freq_feature = []
tf_feature = []
nonlinear_feature = []
for j in range(len(ictal)):
    trans = ictal[j].T
    temp1 = timedomain.timedomain(trans)
    matrix1 = np.array(temp1.time_main(mysteps=5))
    time_feature.append(matrix1)

    temp2 = freqdomain.freqdomain(trans, myfs=fs)
    matrix2 = np.array(temp2.main_freq(percent1=0.5, percent2=0.8, percent3=0.95))
    freq_feature.append(matrix2)

    temp3 = timefreq.timefreq(trans, myfs=fs)
    matrix3 = np.array(temp3.main_tf(smoothwindow=100))
    tf_feature.append(matrix3)

    temp4 = nonlinear.nonlinear(trans, myfs=fs)
    matrix4 = np.array(temp4.nonlinear_main(tau=Tau, m=M, r=R, de=DE, n_perm=4, n_lya=4, band=Band))
    nonlinear_feature.append(matrix4)

time_feature = np.array(time_feature)
freq_feature = np.array(freq_feature)
tf_feature = np.array(tf_feature)
nonlinear_feature = np.array(nonlinear_feature)

file3= 'dataset\\bonn_feature\\ictal_feature.npz'
np.savez(file3,time=time_feature,freq=freq_feature,tf = tf_feature,entropy = nonlinear_feature)

print('finished')
