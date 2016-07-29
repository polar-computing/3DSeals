# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:14:00 2016

@author: Starship
"""

from __future__ import division
import scipy
import pylab
import numpy as np
import math
import shapely.geometry as sg

def polygon_rotator (points, angle=2*math.pi/360):
    '''
    input: points from findContour, sg.polygon from points, and the amount of 
    rotation. 360 degrees = 2*math.pi/360    
    
    requires shapely.geometry, math, numpy as np, scipy, pylab (for plotting)
    adapted from http://gis.stackexchange.com/questions/23587/how-do-i-rotate-
    the-polygon-about-an-anchor-point-using-python-script
    '''    
    #Make the polygon
    poly = sg.Polygon(points)
    
    #Find the centroid (writes a string)    
    cent = poly.centroid.wkt 
    cent_x = np.float64(cent[7:np.uint8((len(cent)+6)/2)]) #pull x as float
    cent_y = np.float64(cent[np.uint8((  len(cent)+7)/2):
                                         len(cent)-1]) #(pull y as float)
    cent_xy = [cent_x, cent_y]
    print cent_xy

    rotation = []
    #pylab.plot(*points.T, lw=5, color='k')
    for angl in scipy.arange(0,2*math.pi, angle): #rotate for every new angle 
        cos = scipy.cos(angl)
        sin = scipy.sin(angl)
        ots = scipy.dot(points-cent_xy,scipy.array([[cos,sin],
                                        [-sin,cos]]))+cent_xy
        rotation.append(ots)    
        pylab.plot(*ots.T)        
            
    pylab.axis('image')
    pylab.grid(True)
    pylab.title('Rotate2D about a point')
    pylab.show()    
    
    return rotation

#rotation = polygon_rotator(points1, angle=2*math.pi/45)