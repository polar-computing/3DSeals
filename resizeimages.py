# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Created on Mon Jul 18 16:29:43 2016

@author: Starship

resize all images in a directoryu to the basewidth specified and write them to
 a directory
"""


####################
# IMAGE RESIZING LOOP
####################

import os

os.chdir('C:/Users/Starship/Documents/GitHub/3DSeals/')


##########
# Resizing of images to 600 px base
##########

import PIL
from PIL import Image
import sys
import os

basewidth = 600


for filename in os.listdir('images0'):
    img = Image.open(r'C:/Users/Starship/Documents/GitHub/3DSeals/images0/'+filename)
    _imgFilename = filename
    _imgFileExtension = os.path.splitext(os.path.basename(filename))[1]    
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    if not os.path.isdir('resized'):
        os.makedirs('resized')
    img.save('C:/Users/Starship/Documents/GitHub/3DSeals/resized/'+str('resized_')+_imgFilename)