# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:48:40 2016
@author: Bento Gonçalves

This code is the working script for Weddell seal photo-ID.
It pre-processes images using gamma correction and histogram equalization.
It hands a binary image based on the most complex contouring image of multiple
gamma values to a watershed segmentation algorithm which erodes the image 
slightly, finds segments based on contours, excludes those that are too small
or too large, and outputs the original image with segments labeled and 
overlaid. 

Incorporated from the GitRepo (https://github.com/polar-computing/3DSeals) are:
gamma_adjust.py
gamma_search.py
gamma_edges.py

"""

import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.color import rgb2gray
from skimage.measure import perimeter



# Watershed segmentation ---- WORKS BEST ON OPENCV 3.1.0


def perim_func (arg1):
    """
    Calculates 'patch complexity', based on the ratio of patch perimeter to patch
    area; 'arg1' must be cv2.findContours output.
    """
    perim = 0
    for i in range(len(arg1)-1):
        dist = math.hypot(arg1[i+1][0,0] - arg1[i][0,0], arg1[i+1][0,1] - arg1[i][0,1] )
        perim = perim + dist
    return perim
 
def gamma(img, gamma = 0.5):
    img_float = np.float32(img)
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
def gamma_search(img, invert=False, save_img=False):
    """
    Gamma correction function (from http://stackoverflow.com/questions/11211260/gamma-correction-power-law-transformation)
    """
    trash = img[:].copy()

    # Generate a range of gamma levels to test against
    gamma_values = np.arange(0.1, 5.1, 0.1)
    output_images = []
    edge_images = []

    # Create directory to save images
    folder_name = "output"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for gamma_level, index in zip(gamma_values, range(gamma_values.size)):
        gamma_data = gamma(image, gamma_level)                                                                       #gamma
        histogram_data = cv2.equalizeHist(src=gamma_data, dst=trash)                                                      #histogram
        if invert:
            threshold_data = cv2.adaptiveThreshold(histogram_data, 240, 
                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 201, 2) #Inverted binary
        else:
            threshold_data = cv2.adaptiveThreshold(histogram_data, 240,
                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 201, 2) #Normal binary
        output_images.append(threshold_data)
        file_name = "image_{0:03d}.jpg".format(index)
        # run edge detection
        edge_data = cv2.Canny(threshold_data, 0, 255)
        # get edge pixels
        edge_images.append(edge_data)
        if save_img: #save in output folder
            cv2.imwrite(os.path.join(folder_name, file_name), threshold_data) 

    max_pixels = 0
    max_gamma = None
    max_index = 0

    for edge_data, index in zip(edge_images, range(len(edge_images))):
        pixel_sum = np.sum(edge_data)
        if pixel_sum > max_pixels :
            max_pixels = pixel_sum
            max_gamma = edge_data
            max_index = index
    return output_images[max_index]

def watershed_seg(img, plot_comp=True, show_img=True, file_name="segmented_seal.png"):
    """
    Segments an image and returns the contours of every patch along with their
    complexity scores and their labels to locate specific patches on the image.
    The segmented seal image is saved on the workspace with the name specified
    on 'file_name' argument. 
    """
    # check input!
    img_copy = img[:].copy()
    # gamma correction
    img_copy2 = img[:].copy()
    img = gamma_search(img)
    
        
    cv2.imshow("gamma corrected image", neg)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #D = ndimage.distance_transform_edt(img)
    D = cv2.erode(img, None, iterations=1)
    localMax = peak_local_max(D, indices=False, min_distance=30,
    labels=img)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=img)
    
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    # loop over the unique labels returned by the Watershed
    # algorithm
    tags = []
    splotches_cont = []
    complexity = []
    # tag will save the tag number
    tag = 0
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        
        if label == 0:
            continue # continue skips this run in the for loop without breaking
    
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(img.shape, dtype="uint8")
        mask[labels == label] = 255
    
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
       
        # draw a circle enclosing the object
        # exclude patches that are too large or too small
        # size threshold is based on the contour area
        area = cv2.contourArea(c)
        if area > (img.size / 10000) and area < (img.size / 100):
            x1,y1,w,h = cv2.boundingRect(c)
            # remove white splotches
            if img_copy[y1:(y1+h), x1:(x1+w)].mean() < 170:
                tag += 1
                splotches_cont.append([point[0] for point in c][0])
                tags.append(tag)
                complexity.append(perim_func(c))
                if show_img:
                    cv2.drawContours(img_copy2, [c], -1, (255, 0, 0), 2)
                    cv2.putText(img_copy2, "#{}".format(tag), (int(x1) - 10, int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)    
    print("[INFO] {} segments retained:".format(tag)) 
            
    if show_img:
        # show the output image 
        cv2.imshow("Output", img_copy2)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if plot_comp:        
        # plotting complexity histogram
        import seaborn as sns
    
        sns.set(rc={"figure.figsize": (18, 9)}, font_scale=3 )
        comp_dist = sns.distplot(complexity, bins=15, axlabel="patch complexity")
    # write it into a file
    cv2.imwrite(file_name, img_copy2)
    # returns contours and their complexity scores
    return(splotches_cont, complexity, tags)

# imread does not crash with invalid input
image = cv2.imread("test1.jpg", 0)
out1 = watershed_seg(image, file_name="eroded_seal.png")