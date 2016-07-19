# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:48:40 2016

@author: Bento Gonçalves
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
    
def gamma_search(image):
    """
    Gamma correction function (from http://stackoverflow.com/questions/11211260/gamma-correction-power-law-transformation)
    """
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
        #cv2.imwrite(os.path.join(folder_name, file_name), threshold_data) #save in output folder

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

def watershed_seg(img, plot=True, file_name="segmented_seal.png"):
    """
    Segments an image and returns the contours of every patch along with their
    complexity scores. Saves a segmented seal image on the workspace  
    """
    img_copy = img[:].copy()
    #gamma correction
    img = gamma_search(img)
    
    D = ndimage.distance_transform_edt(img)
    localMax = peak_local_max(D, indices=False, min_distance=30,
    labels=img)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=img)
    
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    # loop over the unique labels returned by the Watershed
    # algorithm
    splotches_rect = []
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
        if r > (image.size ** 0.24) and r < (image.size ** 0.32):
            tag = tag + 1
            x1,y1,w,h = cv2.boundingRect(c)
            # remove white splotches
            if img_copy[y1:(y1+h), x1:(x1+w)].mean() < 150:
                splotches_rect.append([tag, img_copy[y1:(y1+h), x1:(x1+w)]])
                splotches_cont.append(c)
                complexity.append(perm_func(c))
                if plot:
                    cv2.drawContours(image, [c], -1, (255, 0, 0), 3)
                    cv2.putText(image, "#{}".format(tag), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)             
            
    if plot:
        # show the output image
        cv2.imshow("Output", image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plotting complexity histogram
        import seaborn as sns
    
        sns.set(rc={"figure.figsize": (18, 9)}, font_scale=3 )
        comp_dist = sns.distplot(complexity, bins=15, axlabel="patch complexity")
    # write it into a file
    cv2.imwrite(file_name, image)
    # returns contours and their complexity scores
    return(splotches_cont, complexity)

out = watershed_seg(output)
