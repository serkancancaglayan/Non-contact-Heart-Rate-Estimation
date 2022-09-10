import cv2 as cv
import dlib as dl
import numpy as np
from imutils import face_utils


faceDetectorModelConfigs = ['./cascades/deploy.prototxt.txt', './cascades/res10_300x300_ssd_iter_140000.caffemodel']
shapePredictorModelConfig = './cascades/shape_predictor_68_face_landmarks.dat'

faceDetectorModel = cv.dnn.readNetFromCaffe(faceDetectorModelConfigs[0], faceDetectorModelConfigs[1])
availableGPU = cv.cuda.getCudaEnabledDeviceCount()
if availableGPU > 0:
    faceDetectorModel.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    faceDetectorModel.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

shapePredictorModel = dl.shape_predictor(shapePredictorModelConfig)


def detectFace(frame, drawRect = 1):
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
    return (startX, startY, endX, endY)


def detectLandmarks(frame, startX, startY, endX, endY):
    faceRect = dl.rectangle(startX, startY, endX, endY)
    facialLandmarks = shapePredictorModel(frame, faceRect)
    facialLandmarks = face_utils.shape_to_np(facialLandmarks)
    return facialLandmarks


def extractROIs(frame, facialLandmarks, startY,  drawRect = 1):
    roiRightCheek = frame[facialLandmarks[29][1]:facialLandmarks[33][1], facialLandmarks[54][0]:facialLandmarks[12][0]]
    roiLeftCheek = frame[facialLandmarks[29][1]:facialLandmarks[33][1], facialLandmarks[4][0]:facialLandmarks[48][0]]
    roiForeHead = frame[startY + 20:facialLandmarks[19][1],  facialLandmarks[18][0]:facialLandmarks[25][0]]

    if drawRect:
        cv.rectangle(frame, (facialLandmarks[54][0], facialLandmarks[29][1]), (facialLandmarks[12][0], facialLandmarks[33][1]), (255, 255, 255), 1)
        cv.rectangle(frame, (facialLandmarks[4][0], facialLandmarks[29][1]), (facialLandmarks[48][0], facialLandmarks[33][1]), (255, 255, 255), 1)
        cv.rectangle(frame, (facialLandmarks[18][0], startY + 20), (facialLandmarks[25][0], facialLandmarks[19][1]), (255, 255, 255), 1)
    
    return roiRightCheek, roiLeftCheek, roiForeHead

def extractGreenChannel(frame):
    _, G, _ = cv.split(frame)
    return G