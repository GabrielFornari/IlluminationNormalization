import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
import math

# The following methodology is based on the article entitled
#   "Illumination Compensation and Normalization for
#    Robust Face Recognition Using Discrete Cosine
#    Transform in Logarithm Domain", 2006
#
# Usage:
#   normIllumination(img, DCTdistance = 10, dataType = "float32", base = 2):
#
# Input: uint8 M x N matrix (grayscale 8 bit image)
# Output: real M x N matrix (values between [0, 1])
#
# Parameters:
#   DCTdistance: integer (less equal to one is equivalent to a log transform)
#   dataType: "float32" or "float64"
#   base: real > 1
def normIllumination(img, DCTdistance = 10, dataType = "float32", base = 2):
    # 1. Log transform
    lut = np.asarray([math.log(i, base) for i in (np.arange(0, 256) + 1)], dataType)
    img_log = cv.LUT(img, lut)

    # 2. DCT transform
    DCTCoeficient = cv.dct(img_log)

    # 2.1. Remove DCT coeficients
    if(DCTdistance > 1):
        indices = np.array([[j, i-j] for i in np.arange(1, DCTdistance) for j in np.arange(i+1)])
        DCTCoeficient[indices.T[0], indices.T[1]] = 0

    # 3. IDCT transform
    img_IDCT = cv.idct(DCTCoeficient)

    # 3.1. Normalizing values between [0, 1]
    imgMax = np.max(img_IDCT)
    imgMin = np.min(img_IDCT)
    img_IDCT -= imgMin
    img_IDCT /= (imgMax-imgMin)

    return img_IDCT
