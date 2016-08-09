# -*- coding: utf-8 -*-
"""
Exract light/dark regions from seals
Usage: regions(img)

Images can be most image filetypes.

Now writing as tif to avoid compression. Can remove threshold step.

Segments images sequentially and classifies, then masks to foreground
Classified + masked images are saved in a subdirectory of the input folder

If masks are reversed (ie the seal is masked out), change line 108 from 
-result to result (misc.imsave("classified\\" + imgPath, -result))
"""
"""
1) Segment the seal into broad regions of similarity. IE an area which is
brighter (lighting conditions or just pelage coloration) is segmented as one
area.

2)Create masked images with just these regions

3) Run the rest of the normal pre-process (gamma, binary)

4) Extract patches from that region (retain spatial data - how big is patch?)

Later: work in "where on the seal" solution




Created on Mon Aug 08 16:03:43 2016

@author: Starship
"""

import cv2
import numpy as np
from gamma_adjust import gamma

#blurs the image (adjust with ksize), then erodes (adjust iterations)
#
def regions(img):
    '''
    CURRENTLY (6pm 8 Aug): 
    
    To fix: ksize (and maybe iterations) based on big image. Need to make it work for resized
            or else adaptive to img size. Maybe compare current ksize to length
            original (non-resized vals ksize1=15, iterations=30, ksize=41)
            #update: reduced values, still not adaptive
            
            #Also: thresh value in threshold also not adaptive but works for now
    
    '''
    img_copy = img[:].copy()
    #eroded = cv2.erode(img, None, iterations=10)
    gam = gamma(img, 2.2)
    blur = cv2.GaussianBlur(src=gam, dst=img_copy, ksize=(3, 3), sigmaX=0, 
                            sigmaY=0)
    eroded = cv2.dilate(blur, None, iterations=1)
    #gam = gamma(eroded, 2)
    blur2 = cv2.GaussianBlur(src=eroded, dst=img_copy, ksize=(9,9), sigmaX=0,
                             sigmaY=0)
    thresh_val = np.int(np.mean(blur2))
    ret, threshold_data = cv2.threshold(blur2, 50, 255, cv2.THRESH_BINARY)
    #threshold_data = cv2.adaptiveThreshold(blur2, 255,
                                           #cv2.ADAPTIVE_THRESH_MEAN_C,
                                           #cv2.THRESH_BINARY, 301, 2)                         
    #Create two masked images, one that masks out darker areas, one masks light
    boole = np.bool8(threshold_data)
    light_img = boole * img
    dark_img = img * np.uint8(boole == 0)

    return light_img, dark_img


#img = cv2.imread("resized/masked/resized_test4.tif", 0)
#img = cv2.imread("images/test1.jpg", 0)

#light, dark = regions(img)

#cv2.imshow("Output", light)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
