# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 14:42:11 2016

@author: Okokoko
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.color import rgb2gray


#TODO Get image path as argument from command-line
#TODO Check if file exists

# Load image as greyscale
image = cv2.imread("test1.jpg", 0)
img_copy = image[:].copy()
#input_data = cv2.imread("test2.jpg", 0)


# normalize image
var = 0 
#output_data = cv2.normalize(src=input_data, dst=var, norm_type=cv2.NORM_MINMAX, alpha=0, beta=255)
output_data = image
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
 
def auto_canny(image, sigma=0):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

 
adjusted = adjust_gamma(output_data, gamma=2) 


#sobelx = cv2.Sobel(output_data,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(output_data,cv2.CV_64F,0,1,ksize=5)
#laplacian = cv2.Laplacian(output_data,cv2.CV_64F)
#sobeladj = cv2.Sobel(adjusted,cv2.CV_64F,0,1,ksize=25)
canny = auto_canny(output_data)
adpt = cv2.adaptiveThreshold(output_data,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

gray = rgb2gray(image)
adpt = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
D = ndimage.distance_transform_edt(adpt)
localMax = peak_local_max(D, indices=False, min_distance=30,
	labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
# algorithm
splotches = []
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
    if r > 35 and r < 100:
        #cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        x1,y1,w,h = cv2.boundingRect(c)
        # remove white squares
        if img_copy[y1:(y1+h), x1:(x1+w)].mean() < 170:
            splotches.append([label, img_copy[y1:(y1+h), x1:(x1+w)]])
            cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0,0,255),2) 
        

# show the output image
cv2.imshow("Output", image)
cv2.imwrite('seal_with_squares.jpg', image)
cv2.waitKey(0)

#display image
#plt.figure(figsize=(45,15))
#plt.subplot(3,1,1),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,1,2),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,1,3),plt.imshow(sobely,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(60,20))
plt.subplot(3,1,1), plt.imshow(sobely, cmap  = "gray")
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2), plt.imshow(adpt, cmap = "gray")
plt.title('Adaptive Mean'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3), plt.imshow(thresh, cmap = "gray")
plt.title('Threshold (after grayscale transformation)'), plt.xticks([]), plt.yticks([])

cv2.waitKey(0)
cv2.destroyAllWindows()

