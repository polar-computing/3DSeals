# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:21:10 2016

@author: heatherlynch, ChrisCheCastaldo
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as sg

# Define two polygons and compute their percent overlap

def pol2cart(input):
    x = [None] * len(input)
    y = [None] * len(input)
    for i in range(len(input)):
        x[i] = input[i, 0] * np.cos(input[i, 1])
        y[i] = input[i, 0] * np.sin(input[i, 1])
    return(np.array(zip(x, y)))
    
poly1 = np.array(zip(np.hstack(np.random.uniform(.8, 1, 360)), np.linspace(0, 2*math.pi, 360)))
poly2 = np.array(zip(np.hstack(np.random.uniform(.8, 1, 360)), np.linspace(0, 2*math.pi, 360)))

poly3 = sg.Polygon(pol2cart(poly1))
poly4 = sg.Polygon(pol2cart(poly2))

PercentOverlap = poly3.intersection(poly4).area / poly3.union(poly4).area
print PercentOverlap

# Compute the centroid of the two polygons, add code here to shift then around 

a = poly3.centroid.wkt
b = poly3.centroid.wkt

# Now let's allow rotation and expansiona and contraction and use three actual patches

# rotation

ec = 1.2
ro = 1

poly1 = np.array(zip(np.hstack(np.random.uniform(.8, 1, 360)) * ec, np.linspace(0, 2*math.pi, 360) + ro))
poly2 = np.array(zip(np.hstack(np.random.uniform(.8, 1, 360)) * ec, np.linspace(0, 2*math.pi, 360) + ro))






#Need to convert original polygons into polar coordinates
#Here I have just simulated points in polar coordinates directly

polygon1 = np.column_stack((np.random.rand(360, 1),np.linspace(0,2*math.pi,360)))
polygon2 = np.column_stack((np.random.rand(360, 1),np.linspace(0,2*math.pi,360)))

#The code below will be used to center the two polygons
#The simulated polygons are already centered at zero

#center_x1 = np.mean(polygon1[:,0])
#center_y1 = np.mean(polygon1[:,1])

#center_x2 = np.mean(polygon2[:,0])
#center_y2 = np.mean(polygon2[:,1])

#polygon1_shifted = polygon1-[center_x1,center_y1]
#polygon2_shifted = polygon2-[center_x2,center_y2]

