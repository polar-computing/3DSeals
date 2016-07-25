# -*- coding: utf-8 -*-

"""
Created on Mon Jul 21 2016
@author: Alex Borowicz & Bento Gonçalves

back_extract(img)  :   The beginnings of a background extraction function.   
    This function finds tries to find the outer contour of the seal based on
    the blurred and highly gamma-modified image (gamma = 10). It pulls the 
    points of the contour and builds a shapely polygon object and a shapely 
    LinearRing object (which is currently redundant as these seals should 
    always fulfill the requirements of both).
    It returns: the max contour (in case it finds another segment) as largest1
                the second largest contour as largest2
                the polygon as poly
                the linearRing as linring and
                the points as points
                
    It requires the following packages:
        cv2
        numpy as np
        shapely.geometry.polygon (import LinearRing and Polygon from it)
        
    ISSUES: Only finds one segment. If background doesn't wrap around (seal 
    cropped fore + aft, only picks out top or bottom background.)
   
"""
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point, Polygon
from gamma_adjust import gamma

def back_extract (img):
    trash = img[:].copy()    
    eq_img = cv2.equalizeHist(src=img, dst=trash)     
    #if np.mean(gam[0]) < 126:
    #    img = -img
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
    if len(sorteddata) > 1 :   
        largest2 = sorteddata[1][1]
        points2 = np.array([point[0] for point in largest2])
    else: largest2 = np.asarray((0,0))
    #poly = Polygon(points1)
    linring = LinearRing(points1)
    return largest1, largest2, linring, points1 #also poly, points2, gammed

image = cv2.imread("images/test3.jpg", 0) 
# Here we call the function and store the outputs, loading in the raw seal img
largest1, largest2, poly, points1  = back_extract(image)

#Drawing the back_extract contour for visualization
cv2.drawContours(image, [largest1], -1, (255, 0, 0), -1)
if len(largest2) > 2:
    cv2.drawContours(image, [largest2], -1, (255, 0, 0), -1)
cv2.imshow("contour", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# BELOW: Plots the polygon (in this case the LinearRing). Testing only here.
ring = LinearRing(points1)
x, y = ring.xy
#plotsize based on image size
fig = plt.figure(1, figsize=(image.shape[1]/120, image.shape[0]/120), dpi=120)
ax = fig.add_subplot(111)
ax.plot(x,y,color = '#6699cc',alpha=0.7, linewidth=3, solid_capstyle='round',
        zorder=2)
ax.set_title('Polygon')

from PIL import Image, ImageDraw
np.mean(image[-1])
img = Image.new('L', (1333, 429), 0)
ImageDraw.Draw(img).polygon(points1, fill=1)
#mask = np.array(img)
#plt.imshow(mask),plt.show()
