# import the necessary packages
import argparse
import imutils
import cv2
import sys
import os.path

import numpy as np
from matplotlib import pyplot as plt

"""
*** Idea here is simply to convert given points (simulated by section labeled 'Simulate' to create an image that can be processed by 'compare.py'. Code can be integrated into compare.py).

python compare.py -s compare/shape.jpg
"""
# Read in the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="path to the source image")
# ap.add_argument("-t", "--target", required=True, help="path to the target image")
args = vars(ap.parse_args())

# Check that the files exist.
if not os.path.isfile(args['source']):
    print 'Source file', args['source'], 'does not exist.'
    sys.exit()

# Source: read, gray, blur, thresh, contour.
image = cv2.imread(args['source'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# Get the shape of the object and create an empty image.
height, width, dimensions = image.shape
Color = np.zeros((height, width, 3), 'uint8')

# Re-create the image (proof of concept).
for vv in range(0, height):
    for hh in range(0, width):
        if cv2.pointPolygonTest(cnts[0], (hh, vv), False) != -1:
            Color[vv][hh] = [255, 255, 0]

# Show the comparison.
plt.subplot(3, 2, 1)
plt.imshow(image, 'gray')
plt.subplot(3, 2, 2)
plt.imshow(Color, 'gray')

plt.show()
