# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:48:40 2016

@author: Okokoko
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.color import rgb2gray
from skimage.measure import perimeter

# Watershed segmenting

# Load image
image = cv2.imread("test1.jpg", 0)
# create a copy of our image to extract rectangles
img_copy = image[:].copy()

# convert to grayscale
gray = rgb2gray(image)

# adaptive thresholding for distances
adpt_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
              cv2.THRESH_BINARY,11,2)

# binary thresholding 
bin_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

# distance transform
D = ndimage.distance_transform_edt(adpt_thresh)
localMax = peak_local_max(D, indices=False, min_distance=30,
	labels=adpt_thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=bin_thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

def perm_func (arg1):
    """
    Calculates 'patch complexity', based on the ratio of patch perimeter to patch
    area; 'arg1' must be cv2.findContours output.
    """
    perm = 0
    for i in range(len(arg1)-1):
        dist = math.hypot(arg1[i+1][0,0] - arg1[i][0,0], arg1[i+1][0,1] - arg1[i][0,1] )
        perm = perm + dist
    return perm

# loop over the unique labels returned by the Watershed
# algorithm
splotches_rect = []
splotches_cont = []
complexity = []
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue # continue skips this run in the for loop without breaking

    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(image.shape, dtype="uint8")
    mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
   
    # draw a circle enclosing the object
    ((x, y), r) = cv2.minEnclosingCircle(c)
    #exclude patches that are too large or too small
    #TODO = make it size insensible
    if r > 35 and r < 100:
        #cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        x1,y1,w,h = cv2.boundingRect(c)
        # remove white squares
        if img_copy[y1:(y1+h), x1:(x1+w)].mean() < 170:
            splotches_rect.append([label, img_copy[y1:(y1+h), x1:(x1+w)]])
            splotches_cont.append(c)
            complexity.append(perm_func(c))
            cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0,0,255),2) 
            cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
        

# show the output image
cv2.imshow("Output", image)
# write it into a file
cv2.imwrite('seal_with_squares.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plotting complexity histogram

import seaborn as sns

sns.set(rc={"figure.figsize": (18, 9)}, font_scale=3 )
comp_dist = sns.distplot(complexity, bins=15, axlabel="patch complexity")
