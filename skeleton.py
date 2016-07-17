#!/usr/bin/env python2

import cv2

#TODO Get image path as argument from command-line
#TODO Check if file exists

# Load image as greyscale
input_data = cv2.imread('images/test1.jpg', 0)

print input_data.dtype

# normalize image
output_data = cv2.normalize(src=input_data, norm_type=cv2.NORM_MINMAX, alpha=0, beta=255)

#display image
cv2.imshow('image', output_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
