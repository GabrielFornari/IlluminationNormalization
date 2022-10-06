import cv2 as cv
import numpy as np
import math

# The following algorithm is based on the article entitled
#   "Illumination Compensation and Normalization for
#    Robust Face Recognition Using Discrete Cosine
#    Transform in Logarithm Domain", 2006
#
# Usage:
#   normIllumination(img, DCTdistance = 10, imgType = "uint8", dataType = "float32", base = 2):
#
# Input: uint8 M x N matrix (grayscale 8 bit image)
# Output: M x N matrix (integers between [0, 255] for imgType = "uint8")
#         M x N matrix (float32 between [0, 1] for imgType = "float32")
#         M x N matrix (float64 between [0, 1] for imgType = "float64")
#
# Parameters:
#   DCTdistance: sets the distance of DCT coeficients to be removed
#                "integer" (less equal to one no coeficient is removed)
#                
#   imgOutput:   defines the pixel values of the output
#                "uint8" for image values between [0, 255]
#                "float32" or "float64" for image values between [0, 1]
#
#   dataType:     for precision only, 64 bits give less errors during calcultions
#                 "float32" or "float64"
#
#   base:         logarithm base, can be any number greater than 1
#
def normIllumination(img, DCTdistance = 10, imgOutput = "uint8", dataType = "float32", base = 2):
    # 1. Log transform
    lut = np.asarray([math.log(i, base) for i in (np.arange(0, 256) + 1)], dataType)
    img_log = cv.LUT(img, lut)

    # 2. DCT transform
    DCTCoeficients = cv.dct(img_log)

    # 2.1. Remove DCT coeficients
    if(DCTdistance > 1):
        indices = np.array([[j, i-j] for i in np.arange(1, DCTdistance) for j in np.arange(i+1)])
        DCTCoeficients[indices.T[0], indices.T[1]] = 0

    # 3. IDCT transform
    img_IDCT = cv.idct(DCTCoeficients)

    # 3.1. Normalizing values between [0, 1]
    imgMax = np.max(img_IDCT)
    imgMin = np.min(img_IDCT)
    img_IDCT -= imgMin
    img_IDCT /= (imgMax-imgMin)

    if imgOutput == "uint8":
        return (img_IDCT*255).astype("uint8")
    else:
        return img_IDCT.astype(imgOutput)

# Testing
if __name__ == "__main__":
    img = cv.imread("yaleB07_P00A-070E+00.pgm", cv.IMREAD_GRAYSCALE)
    norm_img = normIllumination(img, 5)
    cv.imshow("Original Image", img)
    cv.imshow("Normalized Image", norm_img)
    cv.waitKey(0)
