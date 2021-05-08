import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    imgL = cv.imread('./StereoMatchingTestings/Art/view1.png')
    imgR = cv.imread('./StereoMatchingTestings/Art/view5.png')

    imgL_new = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR_new = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoBM_create(numDisparities=16, blockSize=6)
    #numDisparities must be positive and divisible by 16
    #SADWindowSize must be odd, be within 5..255 and be not larger than image width or height
    disparity = stereo.compute(imgL_new, imgR_new)
    plt.imshow(disparity, 'gray')
    plt.show()