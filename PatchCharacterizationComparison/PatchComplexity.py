# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

points = np.array([
(292.33,393),
(293.06,393.52),
(299.32,393.6),
(301,393.16),
(303.67,393.09),
(305.9,393.51),
(307.92,392.48),
(311.49,392.33),
(312.69,392.33),
(313.44,393.33),
(314.58,393.75),
(315.77,393.33),
(319.4,393.27),
(322.16,395.23),
(319.67,396.52),
(317.25,397.33),
(315.25,396.42),
(313.27,397.27),
(311.73,396.94),
(308.34,396.58),
(305.58,396.25),
(299.5,397.9),
(289.42,395)])                   

def perm_func (arg1):
    perm = 0
    for i in range(len(arg1)-1):
        dist = math.hypot(arg1[i+1,0] - arg1[i,0], arg1[i+1,1] - arg1[i,1])
        perm = perm + dist
    return perm
            
hull = ConvexHull(points)


print perm_func(points)
print perm_func(points[hull.vertices,])
print perm_func(points)/perm_func(points[hull.vertices,])


plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    
    