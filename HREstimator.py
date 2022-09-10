from locale import normalize
from FaceUtilities import *
from SignalProcessing import *
import matplotlib.pyplot as plt

def captureFrames(source, duration):
    frameBuffer = list()
    videoCap = cv.VideoCapture(source)
    samplingRate =  int(videoCap.get(cv.CAP_PROP_FPS))
    for i in range(samplingRate * duration):
        ret, frame = videoCap.read()
        if ret == True:
            frameBuffer.append(frame)
        else:
            continue

        cv.imshow('Capturing Frames...', frame)
        if cv.waitKey(1) == 27:
            break
    videoCap.release()
    cv.destroyAllWindows()
    return frameBuffer, samplingRate


def estimateHR(frameBuffer, samplingRate, plot = 1):
    meansROI1 = []
    meansROI2 = []
    meansROI3 = []
    
    for frame in frameBuffer:
        startX, startY, endX, endY = detectFace(frame)
        facialLandmarks = detectLandmarks(frame, startX, startY, endX, endY)
        roiRightCheek, roitLeftCheek, roiForeHead = extractROIs(frame, facialLandmarks, startY)
       
        mROI1 = np.mean(extractGreenChannel(roiRightCheek))
        mROI2 = np.mean(extractGreenChannel(roitLeftCheek))
        mROI3 = np.mean(extractGreenChannel(roiForeHead))

        meansROI1.append(mROI1)
        meansROI2.append(mROI2)
        meansROI3.append(mROI3)

        cv.imshow('Calculating HR', frame)
        if cv.waitKey(1) == 27:
            break

    meansROI1 = np.array(meansROI1)
    meansROI2 = np.array(meansROI2)
    meansROI3 = np.array(meansROI3)

    normalizedROI1 = zeroCenterNormalization(meansROI1)
    normalizedROI2 = zeroCenterNormalization(meansROI2)
    normalizedROI3 = zeroCenterNormalization(meansROI3)

    medianROI1 = medianFilter(normalizedROI1, k = 3)
    medianROI2 = medianFilter(normalizedROI2, k = 3)
    medianROI3 = medianFilter(normalizedROI3, k = 3)

    bandPassedROI1 = bandPassFilter(medianROI1, samplingRate, 1, 2.75, 3)
    bandPassedROI2 = bandPassFilter(medianROI2, samplingRate, 1, 2.75, 3)
    bandPassedROI3 = bandPassFilter(medianROI3, samplingRate, 1, 2.75, 3)

    allRoiSignals = np.array([bandPassedROI1, bandPassedROI2, bandPassedROI3])
    pcaSignal = PCA_(allRoiSignals, 1)
    
    windowedSignal = hammingWindow(pcaSignal)

    HRRange, powerSpect = getPowerSpectrum(windowedSignal, samplingRate, 1, 2.75)
    HR = int(HRRange[np.argmax(powerSpect)])

    if plot:
        plt.figure(figsize = (15, 7))
        plt.subplot(2, 1, 1), plt.plot(pcaSignal), plt.title('rPPG Signal')
        plt.subplot(2, 1, 2), plt.plot(HRRange, powerSpect), plt.axvspan(HR, HR +3, color = 'r', alpha = 0.2), plt.title('Power Spectrum')
        plt.show()
    return HR

