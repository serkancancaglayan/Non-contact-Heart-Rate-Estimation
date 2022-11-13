import cv2 as cv
from imutils import face_utils
from SignalProcessing import *
from matplotlib import pyplot as plt


faceDetectorModelConfigs = ['./cascades/deploy.prototxt.txt', './cascades/res10_300x300_ssd_iter_140000.caffemodel']

faceDetectorModel = cv.dnn.readNetFromCaffe(faceDetectorModelConfigs[0], faceDetectorModelConfigs[1])
availableGPU = cv.cuda.getCudaEnabledDeviceCount()
if availableGPU > 0:
    faceDetectorModel.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    faceDetectorModel.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)



def detectFace(frame, drawRect=1):
    (h, w) = frame.shape[:2]
    resizedFrame = cv.resize(frame, (300, 300))
    blob = cv.dnn.blobFromImage(resizedFrame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceDetectorModel.setInput(blob)
    detections = faceDetectorModel.forward()

    detectionConfs = []
    detectionBoxes = []
    for i in range(detections.shape[2]):
        currentConf = detections[0, 0, i, 2]
        currentBox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        detectionConfs.append(currentConf)
        detectionBoxes.append(currentBox.astype("int16"))

    maxConfIdx = np.argmax(detectionConfs)
    (startX, startY, endX, endY) = detectionBoxes[maxConfIdx]
    if drawRect:
        cv.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 1)
    return startX, startY, endX, endY



def sliceFace(frame, startX, startY, endX, endY, marginScale=10):
    faceW = abs(endX - startX)
    faceH = abs(endY - startY)

    adjustedH = faceH - (faceH % marginScale)
    adjustedW = faceW - (faceW % marginScale)

    marginScaleW = adjustedW // marginScale
    marginScaleH = adjustedH // marginScale

    face = frame[startY:endY, startX:endX]
    slices = list()
    for i in range(0, adjustedW, marginScaleW):
        for j in range(0, adjustedH, marginScaleH):
            slice = face[j:j + marginScaleH, i:i + marginScaleW]
            slices.append(np.mean(extractGreenChannel(slice)))
            cv.rectangle(face, (i, j), (i + marginScaleW, j + marginScaleH), (255, 255, 255), 1)
    return slices


def extractGreenChannel(frame):
    _, G, _ = cv.split(frame)
    return G


