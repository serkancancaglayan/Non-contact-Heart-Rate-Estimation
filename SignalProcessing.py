import math
import numpy as np
from scipy.signal import lfilter, butter
from sklearn.decomposition import PCA

def nextPow2(x):
    if x == 0:
        return 1
    return 2 ** math.ceil(math.log2(x))

def zeroCenterNormalization(signal):
    normalizedSignal = (signal - np.mean(signal)) / np.std(signal)
    return normalizedSignal


def medianFilter(signal, k):

    l = (k - 1) // 2
    filteredSignal = np.zeros ((len(signal), k), dtype=signal.dtype)
    filteredSignal[:, l] = signal
    for i in range(l):
        j = l - i
        filteredSignal[j:,i] = signal[:-j]
        filteredSignal[:j,i] = signal[0]
        filteredSignal[:-j,-(i+1)] = signal[j:]
        filteredSignal[-j:,-(i+1)] = signal[-1]

    filteredSignal = np.median(filteredSignal, axis = 1)
    return filteredSignal


def bandPassFilter(signal, samplingRate, lowend, highend, order):
    nyq = 0.5 * samplingRate
    low = lowend / nyq
    high = highend / nyq
    b, a = butter(order, [low, high], btype = 'band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def PCA_(signals, n_components):
    signalsT = np.transpose(signals)
    pca = PCA(n_components = n_components)
    pca.fit(signalsT)

    final_spect = pca.transform(signalsT)
    final_spect = np.squeeze(final_spect, axis =(1,))
    return final_spect

def hammingWindow(signal):
    window = np.hamming(len(signal))
    windowed_signal = np.multiply(signal, window)
    return windowed_signal

def getPowerSpectrum(signal, samplingRate, freqInterestRangeL, freqInterestRangeH):
    numberOfSamples = len(signal)
    NFFT = nextPow2(numberOfSamples * samplingRate)
    freqDomain = np.fft.fft(signal, NFFT)
    conjFreqDomain = np.conjugate(freqDomain)

    powerSpectrum = np.multiply(freqDomain, conjFreqDomain) / NFFT
    powerSpectrum = np.real(powerSpectrum)
    powerSpectrum = powerSpectrum / max(powerSpectrum)

    freqs = np.linspace(0, samplingRate, NFFT)
    fRange = list()
    for i in range(len(freqs)):
        if freqs[i] > freqInterestRangeL and freqs[i] < freqInterestRangeH:
            fRange.append(i + 1)
    fRange = np.array(fRange)

    HRRange = list()
    for i in range(len(fRange)):
        HRRange.append(60 * freqs[fRange[i]])
    HRRange = np.array(HRRange)

    return HRRange, powerSpectrum[fRange]

