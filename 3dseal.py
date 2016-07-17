# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 14:42:11 2016

@author: Okokoko
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc



#TODO Get image path as argument from command-line
#TODO Check if file exists

# Load image as greyscale
input_data = cv2.imread("test1.jpg", 0)



# normalize image
var = 0 
output_data = cv2.normalize(src=input_data, dst=var, norm_type=cv2.NORM_MINMAX, alpha=0, beta=255)
sobelx = cv2.Sobel(output_data,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(output_data,cv2.CV_64F,0,1,ksize=5)
laplacian = cv2.Laplacian(output_data,cv2.CV_64F)



#display image
plt.figure(figsize=(45,15))
plt.subplot(3,1,1),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])


cv2.waitKey(0)
cv2.destroyAllWindows()