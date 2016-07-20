# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:42:09 2016

@author: Starship

Gamma correction function (written by someone else). 

gamma(image, gamma)

image : the image path or object
gamma : a value from 0-5
"""

import numpy as np
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
