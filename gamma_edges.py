# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:56:21 2016

@author: Starship

Takes edge_data from gamma_search.py (not coded in here), and finds the image which
is has the most edge pixels. Returns only this image
"""

#### Detecting gamma looped edges (need the length)

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Is there more edge? #edges is the stacked canny edge arrays
#Takes total array, divides by 19 (19 different gamma corrections)
# Finds image with greatest length

#Add path to big array of all gamma adjusted arrays HERE
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
