# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 08:17:55 2016

@author: Starship

The plan:
Constellations: Blurs image, segments based on blur, finds pattern in original 
                image that falls within segment space.
                
Insert     
    img_copy3 = img[:].copy()
    img = constellations(img)
directly after back_extract call in the watershed script 
(input is back_extract output)
"""

import cv2
import numpy as np

img = cv2.imread("images/test1.jpg", 0)
constellations(image)

#blurs the image (adjust with ksize), then erodes (adjust iterations)
#
def constellations(img):
    img_copy = img[:].copy()
    blur = cv2.GaussianBlur(src=img, dst=img_copy, ksize=(15,15), sigmaX=0, 
                            sigmaY=0)
    eroded = cv2.dilate(blur, None, iterations=5)
    return eroded
    
cv2.imshow("Output", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
