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
    
    Requires: cv2, numpy as np, gamma_adjust (local tool on git)
"""
import cv2
import numpy as np
from gamma_adjust import gamma

def back_extract (img):

    trash = img[:].copy()    
    eq_img = cv2.equalizeHist(src=img, dst=trash)
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
    blank = np.zeros(shape = img.shape)
    if len(points2) > 2 : #If there're two segments
        filled = cv2.fillPoly(blank, [points1, points2], 1)
    else:
        filled = cv2.fillPoly(blank, [points1], 1)
    boole = -np.bool8(filled) #inverts so background is 0
    boole = np.uint8(boole)
    masked = img*boole
    return masked