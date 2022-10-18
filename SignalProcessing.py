import math
import numpy as np
from scipy.signal import lfilter, butter
from sklearn.decomposition import PCA
from skimage.restoration import denoise_wavelet, estimate_sigma

def nextPow2(x):
    if x == 0:
        return 1
    return 2 ** math.ceil(math.log2(x))

def zeroCenterNormalization(signal):
    normalizedSignal = (signal - np.mean(signal)) / np.std(signal)
    return normalizedSignal

def denoiseWavelet(signal):
    sigma_est = estimate_sigma(signal, average_sigmas=True)
    denoised_signal = denoise_wavelet(signal, sigma = sigma_est,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
    return denoised_signal

def bandPassFilter(signal, samplingRate, lowend, highend, order):
    nyq = 0.5 * samplingRate
    low = lowend / nyq
    high = highend / nyq
    b, a = butter(order, [low, high], btype = 'band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def blackmanWindow(signal):
    window = np.blackman(len(signal))
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

