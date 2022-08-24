# coding: utf-8
"""extract nonlinear features from 1*n EEG signal
"""

# Authors: RM Peng <rmpeng19@gmail.com>
import numpy
import utils.psd2 as psd
from scipy import signal, integrate
import pandas as pd

def bin_power(X, Band, Fs):
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.
    Note
    -----
    A real signal can be synthesized, thus not real.
    Parameters
    -----------
    Band
        list
        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.
        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.
     X
        list
        a 1-D real time series.
    Fs
        integer
        the sampling rate in physical frequency
    Returns
    -------
    Power
        list
        spectral power in each frequency bin.
    Power_ratio
        list
        spectral power in each frequency bin normalized by total power in ALL
        frequency bins.
    """

    C = numpy.fft.fft(X)
    C = abs(C)
    Power = numpy.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        c1=int(numpy.floor(Freq / Fs * len(X)))
        c2 = int(numpy.floor(Next_Freq / Fs * len(X)))

        Power[Freq_Index] = sum(C[c1: c2])
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio

def embed_seq(time_series, tau, embedding_dimension):
    """Build a set of embedding sequences from given time series `time_series`
    with lag `tau` and embedding dimension `embedding_dimension`.
    Let time_series = [x(1), x(2), ... , x(N)], then for each i such that
    1 < i <  N - (embedding_dimension - 1) * tau,
    we build an embedding sequence,
    Y(i) = [x(i), x(i + tau), ... , x(i + (embedding_dimension - 1) * tau)].
    All embedding sequences are placed in a matrix Y."""

    if not type(time_series) == numpy.ndarray:
        typed_time_series = numpy.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (
        typed_time_series.size - tau * (embedding_dimension - 1),
        embedding_dimension
    )

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return numpy.lib.stride_tricks.as_strided(
        typed_time_series,
        shape=shape,
        strides=strides
    )

class nonlinear:
    def __init__(self, eegData, myfs):
        #self.data = eegData  # 1*n
        self.data = numpy.squeeze(eegData)
        self.myfs = myfs  # sampling freq

    def ap_entropy(self, M, R):
        X = self.data
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
        A = numpy.tile(Em, (len(Em), 1, 1))
        B = numpy.transpose(A, [1, 0, 2])
        D = numpy.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
        InRange = numpy.max(D, axis=2) <= R

        # Probability that random M-sequences are in range
        Cm = InRange.mean(axis=0)

        # M+1-sequences in range if M-sequences are in range & last values are close
        Dp = numpy.abs(
            numpy.tile(X[M:], (N - M, 1)) - numpy.tile(X[M:], (N - M, 1)).T
        )

        Cmp = numpy.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)

        Phi_m, Phi_mp = numpy.sum(numpy.log(Cm)), numpy.sum(numpy.log(Cmp))

        Ap_En = (Phi_m - Phi_mp) / (N - M)

        return Ap_En

    def permutation_entropy(self, tau,n = 4, ):
        x = self.data
        """Compute Permutation Entropy of a given time series x, specified by
        permutation order n and embedding lag tau.
        Parameters
        ----------
        x
            list
            a time series
        n
            integer
            Permutation order
        tau
            integer
            Embedding lag
        Returns
        ----------
        PE
           float
           permutation entropy
        Notes
        ----------
        Suppose the given time series is X =[x(1),x(2),x(3),...,x(N)].
        We first build embedding matrix Em, of dimension(n*N-n+1),
        such that the ith row of Em is x(i),x(i+1),..x(i+n-1). Hence
        the embedding lag and the embedding dimension are 1 and n
        respectively. We build this matrix from a given time series,
        X, by calling pyEEg function embed_seq(x,1,n).
        We then transform each row of the embedding matrix into
        a new sequence, comprising a set of integers in range of 0,..,n-1.
        The order in which the integers are placed within a row is the
        same as those of the original elements:0 is placed where the smallest
        element of the row was and n-1 replaces the largest element of the row.
        To calculate the Permutation entropy, we calculate the entropy of PeSeq.
        In doing so, we count the number of occurrences of each permutation
        in PeSeq and write it in a sequence, RankMat. We then use this sequence to
        calculate entropy by using Shannon's entropy formula.
        Permutation entropy is usually calculated with n in range of 3 and 7.
        References
        ----------
        Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural
        complexity measure for time series." Physical Review Letters 88.17
        (2002): 174102.
        Examples
        ----------

        # >>> import pyeeg
        # >>> x = [1,2,4,5,12,3,4,5]
        # >>> pyeeg.permutation_entropy(x,5,1)
        2.0"""

        PeSeq = []
        Em = embed_seq(x, tau, n)

        for i in range(0, len(Em)):
            r = []
            z = []

            for j in range(0, len(Em[i])):
                z.append(Em[i][j])

            for j in range(0, len(Em[i])):
                z.sort()
                r.append(z.index(Em[i][j]))
                z[z.index(Em[i][j])] = -1

            PeSeq.append(r)

        RankMat = []

        while len(PeSeq) > 0:
            RankMat.append(PeSeq.count(PeSeq[0]))
            x = PeSeq[0]
            for j in range(0, PeSeq.count(PeSeq[0])):
                PeSeq.pop(PeSeq.index(x))

        RankMat = numpy.array(RankMat)
        RankMat = numpy.true_divide(RankMat, RankMat.sum())
        EntropyMat = numpy.multiply(numpy.log2(RankMat), RankMat)
        PE = -1 * EntropyMat.sum()

        return PE

    def spectral_entropy(self, Band, Power_Ratio=None):
        X = self.data
        Fs = self.myfs
        """Compute spectral entropy of a time series from either two cases below:
        1. X, the time series (default)
        2. Power_Ratio, a list of normalized signal power in a set of frequency
        bins defined in Band (if Power_Ratio is provided, recommended to speed up)
        In case 1, Power_Ratio is computed by bin_power() function.
        Notes
        -----
        To speed up, it is recommended to compute Power_Ratio before calling this
        function because it may also be used by other functions whereas computing
        it here again will slow down.
        Parameters
        ----------
        Band
            list
            boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
            [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
            You can also use range() function of Python to generate equal bins and
            pass the generated list to this function.
            Each element of Band is a physical frequency and shall not exceed the
            Nyquist frequency, i.e., half of sampling frequency.
         X
            list
            a 1-D real time series.
        Fs
            integer
            the sampling rate in physical frequency
        Returns
        -------
        As indicated in return line
        See Also
        --------
        bin_power: pyeeg function that computes spectral power in frequency bins
        """

        if Power_Ratio is None:
            Power, Power_Ratio = bin_power(X, Band, Fs)

        Spectral_Entropy = 0
        for i in range(0, len(Power_Ratio) - 1):
            Spectral_Entropy += Power_Ratio[i] * numpy.log(Power_Ratio[i])
        Spectral_Entropy /= numpy.log(
            len(Power_Ratio)
        )  # to save time, minus one is omitted
        return -1 * Spectral_Entropy

    def svd_entropy(self, Tau, DE, W=None):
        X = self.data
        """Compute SVD Entropy from either two cases below:
        1. a time series X, with lag tau and embedding dimension dE (default)
        2. a list, W, of normalized singular values of a matrix (if W is provided,
        recommend to speed up.)
        If W is None, the function will do as follows to prepare singular spectrum:
            First, computer an embedding matrix from X, Tau and DE using pyeeg
            function embed_seq():
                        M = embed_seq(X, Tau, DE)
            Second, use scipy.linalg function svd to decompose the embedding matrix
            M and obtain a list of singular values:
                        W = svd(M, compute_uv=0)
            At last, normalize W:
                        W /= sum(W)
        Notes
        -------------
        To speed up, it is recommended to compute W before calling this function
        because W may also be used by other functions whereas computing it here
        again will slow down.
        """

        if W is None:
            Y = embed_seq(X, Tau, DE)
            W = numpy.linalg.svd(Y, compute_uv=0)
            W /= sum(W)  # normalize singular values

        return -1 * sum(W * numpy.log(W))

    def samp_entropy(self, M, R):
        X = self.data
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
        A = numpy.tile(Em, (len(Em), 1, 1))
        B = numpy.transpose(A, [1, 0, 2])
        D = numpy.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
        InRange = numpy.max(D, axis=2) <= R
        numpy.fill_diagonal(InRange, 0)  # Don't count self-matches

        Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
        Dp = numpy.abs(
            numpy.tile(X[M:], (N - M, 1)) - numpy.tile(X[M:], (N - M, 1)).T
        )

        Cmp = numpy.logical_and(Dp <= R, InRange[:-1, :-1]).sum(axis=0)

        # Avoid taking log(0)
        Samp_En = numpy.log(numpy.sum(Cm + 1e-100) / numpy.sum(Cmp + 1e-100))

        return Samp_En

    def fisher_info(self, Tau, DE, W=None):
        X = self.data
        if W is None:
            Y = embed_seq(X, Tau, DE)
            W = numpy.linalg.svd(Y, compute_uv=0)
            W /= sum(W)  # normalize singular values
        W_diff=pd.Series(W).diff().dropna().reset_index(drop=True)
        W_shift = pd.Series(W[:-1])
        return sum(W_diff*W_diff/W_shift)

    #largest Lyauponov exponent
    def LLE(self, tau, n, T):
        x = self.data
        fs = self.myfs
        n = 1
        T = 1
        """Calculate largest Lyauponov exponent of a given time series x using
        Rosenstein algorithm.
        Parameters
        ----------
        x
            list
            a time series
        n
            integer
            embedding dimension
        tau
            integer
            Embedding lag
        fs
            integer
            Sampling frequency
        T
            integer
            Mean period, the mean period can be estimated as the reciprocal of the mean frequency in power spectrum
        Returns
        ----------
        Lexp
           float
           Largest Lyapunov Exponent
        Notes
        ----------
        A n-dimensional trajectory is first reconstructed from the observed data by
        use of embedding delay of tau, using pyeeg function, embed_seq(x, tau, n).
        Algorithm then searches for nearest neighbour of each point on the
        reconstructed trajectory; temporal separation of nearest neighbours must be
        greater than mean period of the time series: the mean period can be
        estimated as the reciprocal of the mean frequency in power spectrum
        Each pair of nearest neighbours is assumed to diverge exponentially at a
        rate given by largest Lyapunov exponent. Now having a collection of
        neighbours, a least square fit to the average exponential divergence is
        calculated. The slope of this line gives an accurate estimate of the
        largest Lyapunov exponent.
        References
        ----------
        Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. "A
        practical method for calculating largest Lyapunov exponents from small data
        sets." Physica D: Nonlinear Phenomena 65.1 (1993): 117-134.
        Examples
        ----------
        # >>> import pyeeg
        # >>> X = numpy.array([3,4,1,2,4,51,4,32,24,12,3,45])
        # >>> pyeeg.LLE(X,2,4,1,1)
        # >>> 0.18771136179353307
        """

        Em = embed_seq(x, tau, n)
        M = len(Em)
        A = numpy.tile(Em, (len(Em), 1, 1))
        B = numpy.transpose(A, [1, 0, 2])
        square_dists = (A - B) ** 2  # square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
        D = numpy.sqrt(square_dists[:, :, :].sum(axis=2))  # D[i,j] = ||Em[i]-Em[j]||_2

        # Exclude elements within T of the diagonal
        band = numpy.tri(D.shape[0], k=T) - numpy.tri(D.shape[0], k=-T - 1)
        band[band == 1] = numpy.inf
        neighbors = (D + band).argmin(axis=0)  # nearest neighbors more than T steps away

        # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
        inc = numpy.tile(numpy.arange(M), (M, 1))
        row_inds = (numpy.tile(numpy.arange(M), (M, 1)).T + inc)
        col_inds = (numpy.tile(neighbors, (M, 1)) + inc.T)
        in_bounds = numpy.logical_and(row_inds <= M - 1, col_inds <= M - 1)
        # Uncomment for old (miscounted) version
        # in_bounds = numpy.logical_and(row_inds < M - 1, col_inds < M - 1)
        row_inds[~in_bounds] = 0
        col_inds[~in_bounds] = 0

        # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
        neighbor_dists = numpy.ma.MaskedArray(D[row_inds, col_inds],~in_bounds)
        J = (~neighbor_dists.mask).sum(axis=1)  # number of in-bounds indices by row
        # Set invalid (zero) values to 1; log(1) = 0 so sum is unchanged
        neighbor_dists[neighbor_dists == 0] = 1
        d_ij = numpy.sum(numpy.log(neighbor_dists.data), axis=1)
        mean_d = d_ij[J > 0] / J[J > 0]
        mean_d = (numpy.ones(len(mean_d))) * mean_d[0]

        x = numpy.arange(len(mean_d))

        X = numpy.vstack((x, numpy.ones(len(mean_d)))).T


        [m, c] = numpy.linalg.lstsq(X, mean_d)[0]

        Lexp = fs * m
        return Lexp

    def hurst(self):
        X = self.data
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
        X = numpy.array(X)
        N = X.size
        T = numpy.arange(1, N + 1)
        Y = numpy.cumsum(X)
        Ave_T = Y / T

        S_T = numpy.zeros(N)
        R_T = numpy.zeros(N)

        for i in range(N):
            S_T[i] = numpy.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = numpy.ptp(X_T[:i + 1])

        R_S = R_T / S_T
        R_S = numpy.log(R_S)[1:]
        n = numpy.log(T)[1:]
        A = numpy.column_stack((n, numpy.ones(n.size)))
        [m, c] = numpy.linalg.lstsq(A, R_S)[0]
        H = m
        return H

    def nonlinear_main(self, tau,m,r, de,n_perm =4,n_lya =2,  w=None,band = None,Power_Ratio =None):
        mpf, fmax, fmin, fmin, Ptotal = psd.psd2(self.data,self.myfs)
        avgT = 1/mpf
        # print('avgT',avgT)
        apEn = self.ap_entropy(M =m,R = r)
        premEn = self.permutation_entropy(tau=tau,n=n_perm)
        specEn = self.spectral_entropy(Band=band,Power_Ratio=Power_Ratio)
        svdEn = self.svd_entropy(Tau=tau,DE=de,W = w)

        saEn = self.samp_entropy(M=m,R=r)
        fish = self.fisher_info(Tau=tau,DE=de,W=w)
        lya = self.LLE(tau=tau,n=n_lya,T=avgT)
        hurst = self.hurst()

        entropy_feature = [apEn,premEn,specEn,svdEn,saEn,fish,lya,hurst]
        #entropy_feature = [lya,meand,d]
        return  entropy_feature









