#!/usr/bin/env python2

import cv2
import numpy as np
from matplotlib import pyplot as plt


#gamma correction function (from http://stackoverflow.com/questions/11211260/gamma-correction-power-law-transformation)
def gamma(image,gamma = 0.5):
    img_float = np.float32(image)
    max_pixel = np.max(img_float)
    #image pixel normalisation
    img_normalised = img_float/max_pixel
    #gamma correction exponent calulated
    gamma_corr = np.log(img_normalised)*gamma
    #gamma correction being applied
    gamma_corrected = np.exp(gamma_corr)*255.0
    #conversion to unsigned int 8 bit
    gamma_corrected = np.uint8(gamma_corrected)
    return gamma_corrected

# Load image as greyscale
input_data = cv2.imread('images/test2.jpg', 0)

#gamma correction of greyscale image
gamma_data = gamma(input_data, 4.5)

#equalizes histogram of gamma corr image
equal_data = gamma_data
equal_data = cv2.equalizeHist(src=gamma_data, dst=equal_data)

#############
#Thresholding

thresh_data = cv2.adaptiveThreshold(gamma_data, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 201, 2)
thresh_image = plt.imshow(thresh_data, 'gray', aspect = 'equal')
print thresh_image

##########################

