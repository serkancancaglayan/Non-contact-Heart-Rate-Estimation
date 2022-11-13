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
