import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis
import statistics
from scipy.signal import hilbert, savgol_filter
from pyhht.emd import *
import math
from scipy.stats import kurtosistest, skew
from utils import psd2 as psd


# curve length
def curvelength(Data):
    n_samples = len(Data)
    curvlen = 0
    for i in range(n_samples - 1):
        temp = abs(Data[i + 1] - Data[i])
        curvlen += temp
    return curvlen


# nonlinear energy
def nonlinear_eng(Data):
    n_samples = len(Data)
    total_energy = 0
    for i in range(n_samples - 2):
        nl = abs(Data[i + 1] * Data[i + 1] - Data[i] * Data[i + 2])
        total_energy += nl
    nline_energy = (total_energy / n_samples - 2)
    return nline_energy


# RMS of Amplitude
def rmsa(Data):
    n_samples = len(Data)

    sum = 0
    for i in range(n_samples):
        sum += Data[i] * Data[i]
    return math.sqrt(sum / n_samples)


# num of max and min value
def nummaxmin(Data):
    n_samples = len(Data)
    cnt = 0
    for i in range(n_samples - 1):
        if math.fabs(Data[i] - Data[i + 1]) < 0.01:
            cnt += 1
    return cnt


# zerocrossing
def Zcrossing(Data, step=5):
    th = 0
    cnt = 0
    for cont in range(len(Data) - 1):
        can = Data[cont] * Data[cont + 1]
        can2 = abs(Data[cont] - Data[cont + 1])
        if can < 0 and can2 > th:
            cnt = cnt + 1
    #
    # n_samples = len(Data)
    # cnt = 0
    # for i in range(n_samples - step):
    #     if Data[i] * Data[i+step] < 0:
    #         cnt += 1
    return cnt


# kurt and skew
def kurtskew(Data):
    kurt, kurt_p = kurtosistest(Data)
    skw = skew(Data)
    return kurt, skw


# hjorth
def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """
    activity = np.var(X)
    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return activity, np.sqrt(M2 / TP), np.sqrt(
        float(M4) * TP / M2 / M2
    )  # Hjorth Activity, Mobility and Complexity


# def hjorth(Data):
#     activity = np.var(Data)
#     sigma = np.std(Data)
#     dif1 = np.abs(np.diff(Data))
#     #dif1 = Data.diff().dropna()
#     mobility = dif1.std(ddof = 0) / sigma
#     dif2 = np.abs(np.diff(Data,))
#     dif2 = Data.diff(periods = 2).dropna()
#     complex = dif2.std(ddof = 0) / dif1.std(ddof= 0)
#     return activity, mobility, complex

# time feature extractor
def time_fea_xtractor(Data, mystep=5):
    cv = curvelength(Data)
    ne = nonlinear_eng(Data)
    rA = rmsa(Data)
    numm = nummaxmin(Data)
    zcr = Zcrossing(Data, step=mystep)
    krt, skw = kurtskew(Data)
    ac, mb, cp = hjorth(Data)
    t_fea = [cv, ne, rA, numm, zcr, krt, skw, ac, mb, cp]
    return t_fea


# frequency feature extractor
def freq_fea_xtractor(Data):
    myfs = len(Data) / 2
    mpf, fmax, fmin, fpcntile, Ptotal = psd.psd2(Data, fs=myfs)
    f_fea = [mpf, fmax, fmin, Ptotal]
    return f_fea


# hurst parameter
def hurst(Data):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.
    Parameters
    ----------
    X
        list
        a time series
    Returns
    -------
    H
        float
        Hurst exponent
    Notes
    --------
    Author of this function is Xin Liu
    Examples
    --------
    # >>> import pyeeg
    # >>> from numpy.random import randn
    # >>> a = randn(4096)
    # >>> pyeeg.hurst(a)
    0.5057444
    """
    X = np.array(Data)
    N = X.size
    T = np.arange(1, N + 1)
    Y = np.cumsum(X)
    Ave_T = Y / T

    S_T = np.zeros(N)
    R_T = np.zeros(N)

    for i in range(N):
        S_T[i] = np.std(X[:i + 1]) + 1

        X_T = Y - T * Ave_T[i]
        R_T[i] = np.ptp(X_T[:i + 1]) + 1

    R_S = R_T / S_T
    R_S = np.log(R_S)[1:]

    n = np.log(T)[1:]
    A = np.column_stack((n, np.ones(n.size)))
    [m, c] = np.linalg.lstsq(A, R_S)[0]
    H = m
    return H


# alias
def embed_seq(time_series, tau, embedding_dimension):
    """Build a set of embedding sequences from given time series `time_series`
    with lag `tau` and embedding dimension `embedding_dimension`.
    Let time_series = [x(1), x(2), ... , x(N)], then for each i such that
    1 < i <  N - (embedding_dimension - 1) * tau,
    we build an embedding sequence,
    Y(i) = [x(i), x(i + tau), ... , x(i + (embedding_dimension - 1) * tau)].
    All embedding sequences are placed in a matrix Y."""

    if not type(time_series) == np.ndarray:
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (
        typed_time_series.size - tau * (embedding_dimension - 1),
        embedding_dimension
    )

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return np.lib.stride_tricks.as_strided(
        typed_time_series,
        shape=shape,
        strides=strides
    )


# Appromix entropy
def ap_entropy(X, M, R):
    """Computer approximate entropy (ApEN) of series X, specified by M and R.

    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1

    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.

    Pick Em[i] as a template, for all j such that 0 < j < N - M + 1, we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k. The probability that a random row in Em matches Em[i] is
    \simga_1^{N-M+1} k[i] / (N - M + 1), thus sum(k)/ (N - M + 1),
    denoted as Cm[i].

    We repeat the same process on Emp and obtained Cmp[i], but here 0<i<N-M
    since the length of each sequence in Emp is M + 1.

    The probability that any two embedding sequences in Em match is then
    sum(Cm)/ (N - M +1 ). We define Phi_m = sum(log(Cm)) / (N - M + 1) and
    Phi_mp = sum(log(Cmp)) / (N - M ).

    And the ApEn is defined as Phi_m - Phi_mp.


    Notes
    -----
    Please be aware that self-match is also counted in ApEn.

    References
    ----------
    Costa M, Goldberger AL, Peng CK, Multiscale entropy analysis of biological
    signals, Physical Review E, 71:021906, 2005

    See also
    --------
    samp_entropy: sample entropy of a time series

    """
    N = len(X)

    Em = embed_seq(X, 1, M)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = np.max(D, axis=2) <= R

    # Probability that random M-sequences are in range
    Cm = InRange.mean(axis=0)

    # M+1-sequences in range if M-sequences are in range & last values are close
    Dp = np.abs(
        np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T
    )

    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)

    Phi_m, Phi_mp = np.sum(np.log(Cm)), np.sum(np.log(Cmp))

    Ap_En = (Phi_m - Phi_mp) / (N - M)

    return Ap_En


# sample entropy
def samp_entropy(Data, M, R):
    X = np.array(Data)
    """Computer sample entropy (SampEn) of series X, specified by M and R.
    SampEn is very close to ApEn.

    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1

    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.

    Pick Em[i] as a template, for all j such that 0 < j < N - M , we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k.

    We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.
    The SampEn is defined as log(sum(Cm)/sum(Cmp))
    References
    ----------
    Costa M, Goldberger AL, Peng C-K, Multiscale entropy analysis of biological
    signals, Physical Review E, 71:021906, 2005
    See also
    --------
    ap_entropy: approximate entropy of a time series
    """

    N = len(X)

    Em = embed_seq(X, 1, M)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = np.max(D, axis=2) <= R
    np.fill_diagonal(InRange, 0)  # Don't count self-matches

    Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
    Dp = np.abs(
        np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T
    )

    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).sum(axis=0)

    # Avoid taking log(0)
    Samp_En = np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100))

    return Samp_En


# nonlinear feature extractor
def nonlinear_fea_xtractor(Data, mym, myr):
    ap_en = ap_entropy(Data, M=mym, R=myr)
    sa_en = samp_entropy(Data, M=mym, R=myr)
    hrst = hurst(Data)
    n_fea = [ap_en, sa_en, hrst]
    return n_fea


def smooth(Data, wlen=11, mode='valid'):
    if wlen % 2 == 0:
        wlen += 1
    asmth = savgol_filter(Data, wlen, 3)
    return asmth


def inst_attributes(Data, fs, smoothwlen=11):
    analytic_signal = hilbert(Data)
    aenv = np.abs(analytic_signal)
    iph = np.unwrap(np.angle(analytic_signal))
    ifreq = np.abs((np.diff(iph, prepend=0) / (2.0 * np.pi) * fs))
    ifreqsmth = smooth(ifreq, wlen=smoothwlen, mode='same')
    return aenv, iph, ifreqsmth


def emd_trace(Data, nimfs=3):
    decomposer = EMD(Data, n_imfs=nimfs)
    imfs = decomposer.decompose()
    mean_amp = decomposer.mean_and_amplitude(Data)
    return imfs, mean_amp[0], mean_amp[3]


def emd_hht(Data, myfs, smoothlength=11, mynimfs=0):
    imfsout, amean, amode = emd_trace(Data, nimfs=mynimfs)
    trenv, triph, trifreq = inst_attributes(Data, fs=myfs, smoothwlen=smoothlength)
    return amean, amode, trenv, triph, trifreq


def comp_moment(feature):
    '''this function computes the moments like mean, standard deviation
    and kutosis of the obtained feature vector'''
    step = int(len(feature) / 2)
    # step = len(feature)
    # variables to be used inside loops
    avg_temp = np.zeros([2])
    stn_dev_temp = np.zeros([2])
    kurto_temp = np.zeros([2])
    for i in range(int(len(feature) / step)):
        avg_temp[i] = np.mean(feature[step * i:step * (i + 1)])
        stn_dev_temp[i] = statistics.stdev(feature[step * i:step * (i + 1)])
        kurto_temp[i] = kurtosis(feature[step * i:step * (i + 1)])
    compnent = [avg_temp[0], avg_temp[1], stn_dev_temp[0], stn_dev_temp[1], kurto_temp[0], kurto_temp[1]]
    return compnent


def district_wavelet(Data):
    # cA, cD = pywt.dwt(self.data, 'db4')
    [a, d1, d2, d3] = pywt.waveWavedec(Data, 'db5', level=3)

    # approximation coefficient
    cop_a = comp_moment(a)

    # d1 coffiecient
    cop_d1 = comp_moment(d1)

    # d2 coffiecient
    cop_d2 = comp_moment(d2)

    # d3 coffiecient
    cop_d3 = comp_moment(d3)

    return np.concatenate((cop_a, cop_d1, cop_d2, cop_d3))


def time_freq_fea_xtractor(Data, winlen=11, nimfs=3):
    Data = np.array(Data)
    myfs = len(Data) / 2
    # amean, amode, trenv, triph, trifreq = emd_hht(Data, myfs, winlen, nimfs)
    tf_fea = district_wavelet(Data)
    return tf_fea


def eeg_fea_extra(Data, m, r, sl=11, nfs=1, mystep=5):
    channel = Data.shape[1]

    all_fea = []
    for ch in range(channel):
        cData = Data[:, ch]
        t_fea = np.asarray(time_fea_xtractor(cData, mystep=mystep))
        f_fea = np.asarray(freq_fea_xtractor(cData))
        nlinear = np.asarray(nonlinear_fea_xtractor(cData, m, r))
        tf_fea = time_freq_fea_xtractor(cData, sl, nfs)
        c_fea = np.concatenate((t_fea, f_fea, nlinear, tf_fea))
        all_fea = np.append(all_fea, c_fea)
        # all_fea = all_fea.append(c_fea)
    return all_fea


# import joblib
#
# data = joblib.load('Dataset/TUSZ/pid_00006904_szr_1776.pkl')
# all_fea = eeg_fea_extra(data['test_eeg'][0], m=10, r=0.3,nfs=173)
# print()
import os
folder1 = 'dataset/Bonn/A_Z'
folder2 = 'dataset/Bonn/B_O'
folder3 = 'dataset/Bonn/C_N'
folder4 = 'dataset/Bonn/D_F'
folder5 = 'dataset/Bonn/E_S'
datasetA = []
for file in os.listdir(folder1):
    f1 = folder1+ '/' + file
    datasetA.append(f1)
datasetA = sorted(datasetA)

datasetB = []
for file in os.listdir(folder2):
    f2 = folder2+ '/' + file
    datasetB.append(f2)
datasetB = sorted(datasetB)

datasetC = []
for file in os.listdir(folder3):
    f3 = folder3+ '/' + file
    datasetC.append(f3)
datasetC = sorted(datasetC)

datasetD = []
for file in os.listdir(folder4):
    f4 = folder4+ '/' + file
    datasetD.append(f4)
datasetD = sorted(datasetD)

datasetE = []
for file in os.listdir(folder5):
    f5 = folder5 + '/'+file
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

from joblib import delayed,Parallel
from tqdm import tqdm
# def sub_process(trans):
#     temp1 = timedomain.timedomain(trans)
#     matrix1 = np.array(temp1.time_main(mysteps=5))
#     temp2 = freqdomain.freqdomain(trans, myfs=fs)
#     matrix2 = np.array(temp2.main_freq(percent1=0.5, percent2=0.8, percent3=0.95))
#     temp3 = timefreq.timefreq(trans, myfs=fs)
#     matrix3 = np.array(temp3.main_tf(smoothwindow=100))
#     temp4 = nonlinear.nonlinear(trans, myfs=fs)
#     matrix4 = np.array(temp4.nonlinear_main(tau=Tau, m=M, r=R, de=DE, n_perm=4, n_lya=40, band=Band))
#     return (matrix1,matrix2,matrix3,matrix4)


def sub_process(Data, m=10, r=0.3, sl=11, nfs=173, mystep=5):
    t_fea = np.asarray(time_fea_xtractor(Data, mystep=mystep))
    f_fea = np.asarray(freq_fea_xtractor(Data))
    nlinear = np.asarray(nonlinear_fea_xtractor(Data, m, r))
    tf_fea = time_freq_fea_xtractor(Data, sl, nfs)
    return (t_fea, f_fea, nlinear, tf_fea)

res=Parallel(n_jobs=1)(delayed(sub_process)(normal[j].squeeze()) for j in tqdm(range(len(normal))))
time_feature = [res[i][0] for i in range(len(res))]
freq_feature = [res[i][1] for i in range(len(res))]
tf_feature = [res[i][2] for i in range(len(res))]
nonlinear_feature = [res[i][3] for i in range(len(res))]
time_feature = np.array(time_feature)
freq_feature = np.array(freq_feature)
tf_feature = np.array(tf_feature)
nonlinear_feature = np.array(nonlinear_feature)
print(time_feature.shape,freq_feature.shape,tf_feature.shape,nonlinear_feature.shape)
file1= 'dataset/bonn_feature/normal_feature_41.npz'
np.savez(file1,time=time_feature,freq=freq_feature,tf = tf_feature,entropy = nonlinear_feature)
print('normal features saved!')
#extract features for inter-ictal states
res=Parallel(n_jobs=20)(delayed(sub_process)(inter[j].squeeze()) for j in tqdm(range(len(inter))))
time_feature = [res[i][0] for i in range(len(res))]
freq_feature = [res[i][1] for i in range(len(res))]
tf_feature = [res[i][2] for i in range(len(res))]
nonlinear_feature = [res[i][3] for i in range(len(res))]
time_feature = np.array(time_feature)
freq_feature = np.array(freq_feature)
tf_feature = np.array(tf_feature)
nonlinear_feature = np.array(nonlinear_feature)
print(time_feature.shape,freq_feature.shape,tf_feature.shape,nonlinear_feature.shape)
file2= 'dataset/bonn_feature/inter_feature_41.npz'
np.savez(file2,time=time_feature,freq=freq_feature,tf = tf_feature,entropy = nonlinear_feature)
print('inter features saved!')
#extract features for ictal states
res=Parallel(n_jobs=20)(delayed(sub_process)(ictal[j].squeeze()) for j in tqdm(range(len(ictal))))
time_feature = [res[i][0] for i in range(len(res))]
freq_feature = [res[i][1] for i in range(len(res))]
tf_feature = [res[i][2] for i in range(len(res))]
nonlinear_feature = [res[i][3] for i in range(len(res))]
time_feature = np.array(time_feature)
freq_feature = np.array(freq_feature)
tf_feature = np.array(tf_feature)
nonlinear_feature = np.array(nonlinear_feature)
print(time_feature.shape,freq_feature.shape,tf_feature.shape,nonlinear_feature.shape)
file3= 'dataset/bonn_feature/ictal_feature_41.npz'
np.savez(file3,time=time_feature,freq=freq_feature,tf = tf_feature,entropy = nonlinear_feature)
print('ictal features saved!')
