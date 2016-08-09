# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 10:27:03 2016

@author: Starship

Repository for image pre-processing functions
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
import shapely.geometry as sg
import scipy
import pylab

from cont2points import cont2points

##### Patch complexity #####
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


##### Gamma Correction ########    
def gamma(img, gamma = 0.5):
    'adjusts gamma level of input image'    
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



###### Foreground extractor #######
def fore_extract (img):
    'Extract the foreground of the image using a blur+findContour method and a grabCut method together.'    
    '''
    Attempts to find the background and turn it black. Equalizes the histogram,
    boosts gamma way up to 15 such that the only contour is the seal (usually),
    blurs, then finds that contour, builds a filled polygon from the point, 
    and then multiplies the (inverted) boolean values by the original image 
    such that the background (black, 0) turns all corresponding background 
    pixels in the original black as well.
    
    Requires: cv2, numpy as np
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trash = gray_img[:].copy()    
    eq_img = cv2.equalizeHist(src=gray_img, dst=trash) #Eq histogram
    gammed = gamma(eq_img, gamma=15) #Boost gamma very high
    blur = gammed #Destination for blur
    #Apply blur to minimize contours
    cv2.GaussianBlur(src=gammed, dst=blur, ksize=(35,35), sigmaX=0, sigmaY=0 )         
    cont = cv2.findContours(blur, cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_SIMPLE)[-2] #Find contours
    areaArray = [] #empty array
    for i, c in enumerate(cont): #Fill array with areas
        area = cv2.contourArea(c)
        areaArray.append(area)
    sorteddata = sorted(zip(areaArray, cont), key = lambda x: x[0], 
                        reverse=True) #Sort the data
    largest1 = sorteddata[0][1] #Find the largest contour
    points1 = np.array([point[0] for point in largest1]) #points for largest
    points2 = [0,0] 
    if len(sorteddata) > 1 : #Some images don't have 2 segments 
        largest2 = sorteddata[1][1] #Find 2nd largest contour
        points2 = np.array([point[0] for point in largest2]) #points for 2nd
    else: largest2 = np.asarray((0,0)) #if no 2nd, empty array
    blank = np.zeros(shape = gray_img.shape)
    if len(points2) > 2 : #If there're two segments
        filled = cv2.fillPoly(blank, [points1, points2], 1) #filled poly 1+2
    else:
        filled = cv2.fillPoly(blank, [points1], 1) #Else just 1st poly filled
    boole = ~np.bool8(filled) #inverts so background is 0
    boole = np.uint8(boole) #convert from bool to integer (1 or 0)
    masked = gray_img*boole #Multiply original img by 1 or 0 to mask poly
       ######## Secondary: GrabCut
    '''
    Secondary foreground extractor, cv2's grabCut. both methods pretty good. 
    Together, quite good
    '''
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64) #This pretty much never changes when
    fgdModel = np.zeros((1,65),np.float64) #using grabCut
    
    #Make a rect around the ROI, or here, the whole image-1pixel since it's 
    #already cropped.
    rect = (0,0,img.shape[1]-1, len(img)-1)
    
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    masked2 = img*mask2[:,:,np.newaxis]
    masked2 = cv2.cvtColor(masked2, cv2.COLOR_BGR2GRAY)
    masked = masked2*boole #Make a mask, add to previous mask

    # Find how much white there is. Integrates into inversion decision later 
    how_mask = masked.size - np.count_nonzero(masked)
    
    cv2.imshow("masked img", masked)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return masked, how_mask
    
def gamma_search(img, invert=False, save_img=False): 
    'Search for image w/ most edges after trying different gamma correct levels'    
    """
    Gamma correction function (from http://stackoverflow.com/questions/11211260
    /gamma-correction-power-law-transformation)
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
        gamma_data = gamma(img, gamma_level)                                             #gamma
        histogram_data = cv2.equalizeHist(src=gamma_data, dst=trash)                     #histogram
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
    
