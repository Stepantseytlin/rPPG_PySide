import cv2 as cv
import numpy as np

def meanFrameVideo(fullFrame : cv.Mat):

    # print(f"got ROI shape {type(fullFrame[0]),type(fullFrame[1]),type(fullFrame[2])}")
    if np.array(fullFrame).ndim>2:
        return np.mean( cv.Mat(fullFrame), axis=(0,1))
    else:
        return np.array(fullFrame)
def meanFrameText(fullFrame : cv.Mat):
    # print(f"got ROI shape in TXT {len(fullFrame)}")
    return fullFrame
MeanFrameMethodDict = {"video" : meanFrameVideo,"txt" : meanFrameText, "cam": meanFrameVideo}