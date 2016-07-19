# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:21:10 2016

@author: heatherlynch
"""

import numpy as np
import matplotlib.pyplot as plt

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

polygon1_x = np.random.rand(30, 1)
polygon1_y = np.random.rand(30, 1)

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
  
