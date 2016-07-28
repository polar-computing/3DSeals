#!/usr/bin/env python2

"""
Created on Tue Jul 19 11:42:09 2016
@author: Sebastian

This function incorporates the gamma_adjust.py function.
It applies different gamma correction values to the same image, from 0.1 to 5.0 in increments of 0.1.
It stores these images to disk and also creates an edge-detection array for each which is used in 
gamma_edges.py to calculate the most "complex," or the seal with the most edge pixels.
Writing to disk can be disable by muting line 66 (cv2.imwrite line).

gamma_search.py and gamma_edges.py are incorporated into watershed.py

gamma_search(image)

image : an image object
"""

import cv2, os
import numpy as np
from matplotlib import pyplot as plt

# Load image as greyscale
input_data = cv2.imread(os.path.join("images","test1.jpg"), 0)

def gamma_search(image):
    # Gamma correction function (from http://stackoverflow.com/questions/11211260/gamma-correction-power-law-transformation)
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

    trash = image[:].copy()

    # Generate a range of gamma levels to test against
    gamma_values = np.arange(0.1, 5.1, 0.1)
    output_images = []
    edge_images = []

    # Create directory to save images
    folder_name = "output"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for gamma_level, index in zip(gamma_values, range(gamma_values.size)):
        gamma_data = gamma(input_data, gamma_level)                                                                       #gamma
        histogram_data = cv2.equalizeHist(src=gamma_data, dst=trash)                                                      #histogram
        # TODO clean up noise
        #blur_data = cv2.bilateralFilter(histogram_data, 15, 75, 75)
        threshold_data = cv2.adaptiveThreshold(histogram_data, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 201, 2) #threshold
        output_images.append(threshold_data)
        file_name = "image_{0:03d}.jpg".format(index)
        # run edge detection
        edge_data = cv2.Canny(threshold_data, 0, 255)
        # get edge pixels
        edge_images.append(edge_data)
        cv2.imwrite(os.path.join(folder_name, file_name), threshold_data) #save in output folder

    max_pixels = 0
    max_gamma = None
    max_index = 0

    for edge_data, index in zip(edge_images, range(len(edge_images))):
        pixel_sum = np.sum(edge_data)
        if pixel_sum > max_pixels :
            max_pixels = pixel_sum
            max_gamma = edge_data
            max_index = index
    print max_index
    return output_images[max_index]

output = gamma_search(input_data)
