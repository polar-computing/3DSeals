# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:56:21 2016

@author: Starship
"""

#### Detecting gamma looped edges (need the length)

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Is there more edge? #edges is the stacked canny edge arrays
#Takes total array, divides by 19 (19 different gamma corrections)
# Finds image with greatest length

#Add path to big array of all gamma adjusted arrays HERE
edge_images = ()


maxi = 0

for i in edge_images:
    edge_sum = sum(i)
    if edge_sum > maxi :
        maxi = edge_sum
        best_gamma = i
        
best_edge = edge_images[best_gamma]

plt.subplot(122),plt.imshow(best_gamma, cmap = 'gray')
plt.title('Best Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()