from FaceUtilities import *
from SignalProcessing import *
import matplotlib.pyplot as plt
from scipy.io import savemat


def captureFrames(source, duration):
    frameBuffer = list()
    videoCap = cv.VideoCapture(source)
    samplingRate = int(videoCap.get(cv.CAP_PROP_FPS))
    print('Sampling Rate :', samplingRate)
    if duration != 0:
        for _ in range(samplingRate * duration):
            ret, frame = videoCap.read()
            if ret:
                frameBuffer.append(frame)
            else:
                continue

            cv.imshow('Capturing Frames...', frame)
            if cv.waitKey(1) == 27:
                break
    else:
        while True:
            ret, frame = videoCap.read()
            if not ret:
                break
            else:
                frameBuffer.append(frame)
            cv.imshow('Capturing Frames...', frame)
            if cv.waitKey(1) == 27:
                break
    videoCap.release()
    cv.destroyAllWindows()
    return frameBuffer, samplingRate


def estimateHR(frameBuffer, samplingRate, plot=1):
    meansROI1 = []
    meansROI2 = []
    meansROI3 = []

    for frame in frameBuffer:
        try:
            startX, startY, endX, endY = detectFace(frame)
            facialLandmarks = detectLandmarks(frame, startX, startY, endX, endY)
            roiRightCheek, roiLeftCheek, roiForeHead = extractROIs(frame, facialLandmarks, startY)

            mROI1 = np.mean(extractGreenChannel(roiRightCheek))
            mROI2 = np.mean(extractGreenChannel(roiLeftCheek))
            mROI3 = np.mean(extractGreenChannel(roiForeHead))

            meansROI1.append(mROI1)
            meansROI2.append(mROI2)
            meansROI3.append(mROI3)
        except:
            meansROI1.append(0)
            meansROI2.append(0)
            meansROI3.append(0)
        cv.imshow('Calculating HR...', frame)
        if cv.waitKey(1) == 27:
            break

    meansROI1 = np.array(meansROI1)
    meansROI2 = np.array(meansROI2)
    meansROI3 = np.array(meansROI3)

    normalizedROI1 = zeroCenterNormalization(meansROI1)
    normalizedROI2 = zeroCenterNormalization(meansROI2)
    normalizedROI3 = zeroCenterNormalization(meansROI3)

    denoisedROI1 = denoiseWavelet(normalizedROI1)
    denoisedROI2 = denoiseWavelet(normalizedROI2)
    denoisedROI3 = denoiseWavelet(normalizedROI3)

    bandPassedROI1 = bandPassFilter(denoisedROI1, samplingRate, 1, 2.75, 3)
    bandPassedROI2 = bandPassFilter(denoisedROI2, samplingRate, 1, 2.75, 3)
    bandPassedROI3 = bandPassFilter(denoisedROI3, samplingRate, 1, 2.75, 3)

    final_psd = getPSD(bandPassedROI1) * getPSD(bandPassedROI2) * getPSD(bandPassedROI3)
    final_signal = np.fft.ifft(final_psd)

    d = {'ppg_signal': final_signal}
    savemat('ppg_signal.mat', d)
    windowedSignal = hammingWindow(final_signal)
    HRRange, powerSpect = getPowerSpectrum(windowedSignal, samplingRate, 1, 2.75)
    HR = int(HRRange[np.argmax(powerSpect)])

    if plot:
        plt.figure(figsize=(15, 7))
        plt.plot(final_signal), plt.title('rPPG Signal')
        plt.figure()
        plt.plot(HRRange, powerSpect), plt.axvspan(HR - 2, HR + 4, color='r', alpha=0.2), plt.title('Power Spectrum')
        plt.show()

    return HR, final_signal


def estimateHRSlices(frameBuffer, samplingRate, plot=1):
    allSliceMeans = list()
    for frame in frameBuffer:
        startX, startY, endX, endY = detectFace(frame)
        allSliceMeans.append(sliceFace(frame, startX, startY, endX, endY))
        cv.imshow("Slice Method", frame)
        if cv.waitKey(1) == 27:
            break
    allSliceMeans = np.transpose(np.array(allSliceMeans))
    bestSlices = getNoiseFreeSlices(allSliceMeans, 10)
    for i in range(len(bestSlices)):
        bestSlices[i] = np.array(bestSlices[i])
        bestSlices[i] = zeroCenterNormalization(bestSlices[i])
        bestSlices[i] = denoiseWavelet(bestSlices[i])
        bestSlices[i] = bandPassFilter(bestSlices[i], samplingRate, 1, 2.75, 3)
    finalPsd = getPSD(bestSlices[0])
    for slice in bestSlices[1:]:
        finalPsd = finalPsd * getPSD(slice)
    finalSignal = np.real(np.fft.ifft(finalPsd))
    windowedSignal = hammingWindow(finalSignal)
    HRRange, powerSpect = getPowerSpectrum(windowedSignal, samplingRate, 1, 2.75)
    HR = int(HRRange[np.argmax(powerSpect)])
    if plot:
        plt.figure(figsize=(15, 7))
        plt.plot(finalSignal), plt.title('rPPG Signal')
        plt.figure()
        plt.plot(HRRange, powerSpect), plt.axvspan(HR - 2, HR + 4, color='r', alpha=0.2), plt.title('Power Spectrum')
        plt.show()

    return HR, finalSignal
