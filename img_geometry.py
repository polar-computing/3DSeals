# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:13:56 2016

@author: Starship
"""
from __future__ import division
import numpy as np
import scipy
import pylab
import math
import shapely.geometry as sg

def cart2pol(x, y):
    """
    converts cartesian coordinates to polar coordinates. 
    Input: x and y of a point
    Output: polar x and y of a point
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    """
    converts polar coordinates to cartesian coordinates
    Input: x and y of a point
    Output: cartesian x and y of a point
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def poly_relocate(polylist):
    """
    Relocates all polygons to be centered on 0,0
    Input: A list of polygons as arrays of points (2L), the output of 
    polygon_rotator found in img_geometry.
    Output: A list of similar shape to the input
    requires Shapely.geometry as sg
             numpy as np
    """
    
    cent = sg.asPolygon(polylist[0]).centroid.wkt #find polygon centroid
    cent_x = np.float64(cent[7:np.uint8((len(cent)+6)/2)]) #pull x as float
    cent_y = np.float64(cent[np.uint8((  len(cent)+7)/2):
                                         len(cent)-1]) #pull y as float

    ## Subtract centroid value from all points in all rotations of the input polygon
    rot_std = []
    for pt in polylist:
        
        rot_stdx = []
        rot_stdy = []
        for ptx, pty in pt:
            rot_stdx.append(ptx-cent_x)
            rot_stdy.append(pty-cent_y)
        rot_std.append(np.asarray(zip(rot_stdx, rot_stdy)))
        #avg_x 
    return(rot_std)

def polygon_rotator (points, angle=2*math.pi/360):
    '''
    input: points from findContour, sg.polygon from points, and the amount of 
    rotation. 360 degrees = 2*math.pi/360    
    
    requires shapely.geometry, math, numpy as np, scipy, pylab (for plotting)
    adapted from http://gis.stackexchange.com/questions/23587/how-do-i-rotate-
    the-polygon-about-an-anchor-point-using-python-script
    
    CURRENT ISSUES: Produces rounding errors such that the centroid is nearly
    (0,0) but not quite. Probably not a big deal given the magnitude of the error
    But still means that identical polys might not line up perfectly which could lead to much smaller 
    '''    
    #Make the polygon
    poly = sg.Polygon(points)
    
    #Find the centroid (writes a string)    
    cent = poly.centroid.wkt.split("(",1)[1] #centroid.wkt outputs a str. Here we cut it down to the coords
    cent = cent.split(")",1)[0] #splitting further here and below
    cent_x = np.float64(cent.split(" ",1)[0])
    cent_y = np.float64(cent.split(" ",1)[1])
    
    cent_xy = [cent_x, cent_y]
    #print cent_xy

    rotation = []
    #pylab.plot(*points.T, lw=5, color='k')
    for angl in scipy.arange(0,2*math.pi, angle): #rotate for every new angle 
        cos = scipy.cos(angl)
        sin = scipy.sin(angl)
        ots = scipy.dot(points-cent_xy,scipy.array([[cos,sin],
                                                    [-sin,cos]]))+cent_xy
        rotation.append(ots) #Make list of arrays of points for each rotation
        #pylab.plot(*ots.T)   #Plot all points. Only for testing/visualization
            
    #pylab.axis('image')
    #pylab.grid(True)
    #pylab.title('Rotate2D about a point')
    #pylab.show()    
    
    return rotation