#!/usr/bin/env python2

import cv2
import numpy as np
from matplotlib import pyplot as plt

#TODO Get image path as argument from command-line
#TODO Check if file exists

# Load image as greyscale
input_data = cv2.imread('images/test1.jpg', 0)

# Gamma correction
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

gamma_data = adjust_gamma(input_data, 0.75)

# Histogram equalization
histogram_data = cv2.equalizeHist(gamma_data)

# Threshold
blur = cv2.GaussianBlur(histogram_data,(5,5),0)
ret3,threshold_data = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# Display image. Press any key to quit
cv2.imshow('image', threshold_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
