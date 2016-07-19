# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:21:10 2016

@author: heatherlynch, ChrisCheCastaldo
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as sg
import shapely.geometry.base as sgb

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

a = poly3.union(poly4)
a.area

from shapely.geometry import Point
>>> a = Point(1, 1).buffer(1.5)
>>> b = Point(2, 1).buffer(1.5)
>>> c = a.intersection(b)
>>> c.area


plt.plot(poly3[:,0], poly3[:,1], 'o')


poly3.union(poly4)


poly3.area
poly4.area


sgb.area(poly3)
sgb.area(poly4)

,poly4)



polygon1 = zip(x,y)

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

polygon1_shifted = polygon1
polygon2_shifted = polygon2

temp=pol2cart(np.random.rand(1, 1)*10,np.linspace(0,2*math.pi,10)[1])
for i in range(10)[1:10]:
    temp=np.row_stack((temp,pol2cart(np.random.rand(1, 1)*10,np.linspace(0,2*math.pi,10)[i])))
    
plt.plot(temp[0][0],temp[1][0],'.')
temp=np.column_stack((temp[0][0],temp[1][0]))

import shapely.geometry

a=shapely.geometry.Polygon(np.column_stack((temp[0][0],temp[1][0])))
  
