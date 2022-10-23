from HREstimator import *
import argparse
from sys import argv

if len(argv) > 0:
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--videoSource', type = str, help = "Webcam source", default = '0')
    argParser.add_argument('--duration', type = int, help = "Video duration", default = 15)
    argParser.add_argument('--plot', type = int, help = "Plot of rPPG Signal", default = 1)
    args = argParser.parse_args()

    source = 0 if args.videoSource == '0' else args.videoSource
    duration = args.duration
    plot = args.plot
    
    frameBuffer, samplingRate = captureFrames(source, duration)
    HR, pca_signal = estimateHRSlices(frameBuffer, samplingRate, plot)
    print('Heart rate : ', HR)

