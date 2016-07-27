# -*- coding: utf-8 -*-

"""
Created on Mon Jul 21 2016
@author: Alex Borowicz & Bento Gonçalves

    Attempts to find the background and turn it black. Equalizes the histogram,
    boosts gamma way up to 15 such that the only contour is the seal (usually),
    finds that contour, builds a filled polygon from the point, and then 
    multiplies the (inverted) boolean values by the original image such that 
    the background (black, 0) turns all corresponding background pixels in the 
    original black as well.
    
    Then runs grabCut, a foreground extractor which is generally better but
    misses small pieces (that would certainly be segmented in findContour) 
    which are smoothed out by the blunt axe of the rest of back_extract
    
    Requires: cv2, numpy as np, gamma_adjust (local tool on git)
"""
import cv2
import numpy as np
from gamma_adjust import gamma
    
img = cv2.imread("images/test1.jpg")    
    
def back_extract (img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trash = gray_img[:].copy()    
    eq_img = cv2.equalizeHist(src=gray_img, dst=trash)
    gammed = gamma(eq_img, gamma=15)
    blur = gammed
    cv2.GaussianBlur(src=gammed, dst=blur, ksize=(35,35), sigmaX=0, sigmaY=0 )         
    cont = cv2.findContours(blur, cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_SIMPLE)[-2]
    areaArray = []
    for i, c in enumerate(cont):
        area = cv2.contourArea(c)
        areaArray.append(area)
    sorteddata = sorted(zip(areaArray, cont), key = lambda x: x[0], 
                        reverse=True)
    largest1 = sorteddata[0][1]
    points1 = np.array([point[0] for point in largest1])
    points2 = [0,0]
    if len(sorteddata) > 1 : #Some images don't have 2 segments 
        largest2 = sorteddata[1][1]
        points2 = np.array([point[0] for point in largest2])
    else: largest2 = np.asarray((0,0))
    blank = np.zeros(shape = gray_img.shape)
    if len(points2) > 2 : #If there're two segments
        filled = cv2.fillPoly(blank, [points1, points2], 1)
    else:
        filled = cv2.fillPoly(blank, [points1], 1)
    boole = ~np.bool8(filled) #inverts so background is 0
    boole = np.uint8(boole)
    masked = gray_img*boole
        ######## Secondary: GrabCut

    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = (0,0,img.shape[1]-1, len(img)-1)
    
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    masked2 = img*mask2[:,:,np.newaxis]
    masked2 = cv2.cvtColor(masked2, cv2.COLOR_BGR2GRAY)
    masked = masked2*boole
    
    # Find how much white there is. Integrates into inversion decision later 
    how_mask = masked.size - np.count_nonzero(masked)
    
    cv2.imshow("masked img", masked)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return masked, how_mask