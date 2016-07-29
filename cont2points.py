# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:39:07 2016

@author: Starship
"""
import numpy as np
import cv2


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
    points = []
    for cnt in sorteddata:
        c = sorteddata[cnt][1]
        pts = np.array([point[0] for point in c])
        points.append(pts)
    return points