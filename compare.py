# import the necessary packages
import argparse
import imutils
import cv2
import sys
import os.path

"""
python compare.py -s compare/shape.jpg -t compare/shape2.jpg
python compare.py -s compare/shape.jpg -t compare/shape2.jpg
python compare.py -s compare/shape.jpg -t compare/shape20.jpg
python compare.py -s compare/shape.jpg -t compare/shape4-larger.jpg
python compare.py -s compare/shape.jpg -t compare/shape4-stretched.jpg
python compare.py -s compare/shape.jpg -t compare/shape5-stretched.jpg  
"""
# Read in the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="path to the source image")
ap.add_argument("-t", "--target", required=True, help="path to the target image")
args = vars(ap.parse_args())

# Check that the files exist.
if not os.path.isfile(args['source']):
    print 'Source file', args['source'], 'does not exist.'
    sys.exit()

if not os.path.isfile(args['target']):
    print 'Target file', args['target'], 'does not exist.'
    sys.exit()

# Source: read, gray, blur, thresh, contour.
image = cv2.imread(args['source'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# Target: read, gray, blur, thresh, contour.
image2 = cv2.imread(args['target'])
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
thresh2 = cv2.threshold(blurred2, 60, 255, cv2.THRESH_BINARY)[1]

cnts2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]

# Compare the source and the target.
ret = cv2.matchShapes(cnts[0], cnts2[0],1,0.0)

# Print results.
print args['source'], ' vs. ', args['target']
print 'result: ', ret
