import argparse
import imutils
import cv2
import sys
import os.path

import numpy as np
from matplotlib import pyplot as plt

"""
- Not ready!
"""

image = cv2.imread('images/test1.jpg')
fgbg = cv2.BackgroundSubtractorMOG(500, 6, 0.9, .1)

fgmask = fgbg.apply(image)

plt.subplot(3, 2, 1)
plt.imshow(fgmask, 'gray')

plt.show()
