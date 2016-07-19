# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:29:43 2016

@author: Starship
"""

#!/usr/bin/env python2
####################
# IMAGE RESIZING LOOP
####################

import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

os.chdir('C:/Users/Starship/Documents/GitHub/3DSeals/')


##########
# Resizing of images to 600 px base
##########

import PIL
from PIL import Image
import sys

basewidth = 600


for filename in os.listdir('images'):
    img = Image.open(r'C:/Users/Starship/Documents/GitHub/3DSeals/images/'+filename)
    _imgFilename = filename
    _imgFileExtension = os.path.splitext(os.path.basename(filename))[1]    
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    if not os.path.isdir('resized'):
        os.makedirs('resized')
    img.save(r'C:/Users/Starship/Documents/GitHub/3DSeals/resized/'+_imgFilename+str('resized')+_imgFileExtension)