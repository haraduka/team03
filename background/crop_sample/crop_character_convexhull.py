# this program can crop only character convexhull position.

import sys

from PIL import Image, ImageOps
import cv2
import numpy as np

args = sys.argv

img = cv2.imread(args[1])

#ret,thresh = cv2.threshold(img,127,255,0)
#contours,hierarchy = cv2.findContours(thresh, 1, 2)
#print(contours[0])

detector = cv2.FeatureDetector_create("FAST")
keypoints = detector.detect(img)
keypoints_pt = np.array([[keypoints[i].pt] for i in range(len(keypoints))]).astype(np.int32)

hull = cv2.convexHull(keypoints_pt, returnPoints=False)
#print([np.array([keypoints_pt[p[0]] for p in hull])])
out = cv2.drawKeypoints(img, [keypoints[p[0]] for p in hull], None)
#out = cv2.drawKeypoints(img, keypoints, None)
cv2.polylines(img, [np.array([keypoints_pt[p[0]] for p in hull])], True, (0, 255, 255))

cv2.imshow("result", img)
cv2.waitKey(0)
