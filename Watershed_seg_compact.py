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
import shapely.geometry as sg
import scipy
import pylab

#LOCAL FUNCTIONS
from cont2points import cont2points
from rotation import polygon_rotator
from region_extract import regions

from image_funs import perim_func, gamma, fore_extract, gamma_search

# Watershed segmentation ---- WORKS BEST ON OPENCV 3.1.0

    
def watershed_seg(img, plot_comp=True, show_img=True, inv=False, file_name="segmented_seal.png"):
    """
    Segments an image and returns the contours of every patch along with their
    complexity scores and their labels to locate specific patches on the image.
    The segmented seal image is saved on the workspace with the name specified
    on 'file_name' argument. 
    """
    #img, how_mask = fore_extract(img)
       
    # check input!
    img_copy = img[:].copy()
    # gamma correction
    img_copy2 = img[:].copy()
    img_copy3 = img[:].copy()
    
    #light, dark = regions(img)    
    darklight = regions(img) #find highlight/shadow regions, extract separately    
    
    tags = [] 
    splotches_cont = []
    complexity = []
    
    for mask in darklight:

        if inv:    
            img = gamma_search(mask, invert=True, save_img=True)
        else:
            img = gamma_search(mask, invert=False, save_img=True)
                
        cv2.imshow("gamma corrected -> binary image", img)  
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
        
        # loop over the unique labels returned by the Watershed algorithm
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
            #cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            #    cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            #print cnts[0]
                                    
            def cont2points(cont):
                '''
                transforms cv2.findContours output to points, arranges by area
                requires numpy as np, cv2
                outputs points corresponding to patches    
                '''
                areaArray = []
                for i, c in enumerate(cont):
                    area = cv2.contourArea(c)
                    areaArray.append(area)
                    sorteddata = sorted(zip(areaArray, cont), key = lambda x: x[0], 
                                        reverse=True)
                    print sorteddata
                points = []
                for con in enumerate(sorteddata):
                    c = sorteddata[con][0]
                    pts = np.array([point[0] for point in c])
                    points.append(pts)
                    points.append([points.append[0]]) #Testing with this
                    return sorteddata       
       
            c = max(cnts, key=cv2.contourArea)
            
            # draw a circle enclosing the object
            # exclude patches that are too large or too small
            # size threshold is based on the contour area
            area = cv2.contourArea(c)
            if area > (img.size / 10000) and area < (img.size / 75):
                x1,y1,w,h = cv2.boundingRect(c)
                # remove white splotches
                if img_copy[y1:(y1+h), x1:(x1+w)].mean() < 170:
                    tag += 1
                    ### TEST#########
                    p_ray = []
                    #p_ray.append(p.array([point[0] for point in c]))
                    p_ray.append([point[0] for point in c])
                    #p_ray.append([c[1][0]])
                    #np.insert(p_ray, len(p_ray), [p_ray[0][0], p_ray[0][1]])
                    splotches_cont.append(np.array([point[0] for point in c]))
                    
                    print np.array(point[0] for point in c)
                    #splotches_cont = p_ray               
                    #splotches_cont[len(splotches_cont)-1].append(np.array(splotches_cont[len(splotches_cont)-1][0]))  #len(splotches_cont[len(splotches_cont)]),              
                    #print c[0][0]
                    #splotches_cont.append(p_ray)
                    tags.append(tag)
                    complexity.append(perim_func(c))   # x = [a1, b2, c3]
                    
                    if show_img:
                        cv2.drawContours(img_copy2, [c], -1, (255, 0, 0), 2)
                        cv2.putText(img_copy2, "#{}".format(tag), (int(x1) - 10, 
                                    int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 0, 255), 2)    
                    #for i in np.arange(0, len(splotches_cont), 1):
                        #    splotches_cont[i].append(splotches_cont[i][0])
        print("[INFO] {} segments retained:".format(tag)) 
            
        if show_img:
            # show the output image

            cv2.imshow("Output Segmented image", img_copy2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if plot_comp:        
            # plotting complexity histogram
            import seaborn as sns

            sns.set(rc={"figure.figsize": (18, 9)}, font_scale=3 )
            comp_dist = sns.distplot(complexity, bins=15, 
                                     axlabel="patch complexity")
            # write it into a file
            cv2.imwrite(file_name, img_copy2)
    # returns contours and their complexity scores
    return(splotches_cont, complexity, tags)

# imread does not crash with invalid input
image = cv2.imread("resized_seal1/masked/resized_seal1_1.tif", 0) #add ,0 for grayscale (currently interferes with back_extract, but fine for Phil's when implemented I think)
out1 = watershed_seg(image, show_img=True, file_name="eroded_seal.png")

out1[0][0][0]
splotches = out1[0]

########## Working out overlap/rotation for matching

rots = polygon_rotator(out1[0][16], angle=2*math.pi/180) #test1 eroded
rots_1 = polygon_rotator(out2[0][20], angle=2*math.pi/4) #seal1_1 eroded1
sg.Polygon(rots[18]).is_valid
out2 = out1 #test1

PercentOverlap = poly1.intersection(poly2).area / poly1.union(poly2).area
print PercentOverlap

poly1 = sg.asPolygon(out1[0][0])

poly1 = sg.asPolygon(rots[0])
poly2 = sg.asPolygon(rots_1[0]) #pol2cart(rots_1[18])
poly2 = poly1

poly3 = sg.Polygon(pol2cart(rots[18]))
poly4 = sg.Polygon(pol2cart(rots_1[20]))
poly3.buffer(0).wkt
poly4 = poly3.buffer(0)
poly1
poly2
PercentOverlap = poly1.intersection(poly2).area / poly1.union(poly2).area
poly1.type
poly1.is_valid

### Why does pol2cart fuck everything up?
### Do subsequent rotations in poly rotator overlay something? look at poly plotted
### How to close polygons!? (sg.polygon already handles that just fine so not a big concern)

len(rots[3])
cartcheck = pol2cart(rots[0])
cartcheck = pol2cart(rots[0][0][0],rots[0][0][1])
cartcheck = []
for pointx, pointy in rots[0]:
    cartcheck.append(pol2cart(pointx, pointy))

rots[0][0][0],rots[0][0][1]
sg.Polygon(rots[170])
sg.Polygon(cartcheck)
range(len(rots[4]))
def pol2cart(input):
    cart = []
    x = [None] * len(input)
    y = [None] * len(input)
    for i in range(len(input)):
        x = input[i][0] * np.cos(input[i][1])
        y = input[i][0] * np.sin(input[i][1])
        cart.append((x,y))
    return(cart)
    #return(np.array(zip(x, y)))


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

x = [None] * len(out1[0][0])

len(out1)