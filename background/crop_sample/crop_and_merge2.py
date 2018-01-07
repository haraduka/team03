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

base_color = img[0][0]
print(base_color)
height, width, channels = img.shape
print img.shape
center_x = center_y = point_count = 0
for y in range(height):
    for x in range(width):
        if np.sum(np.abs(base_color - img[y][x])) > 2:
            center_x += x
            center_y += y
            point_count += 1
center_x = center_x/point_count
center_y = center_y/point_count

dst_img = np.ones((height, width, channels), np.uint8)
for y in range(height):
    for x in range(width):
        dst_img[y][x] = base_color

ite_num = 10

while ite_num:
    ite_num -= 1
    ry = np.random.randint(height)
    rx = np.random.randint(width)
    for y in range(height):
        for x in range(width):
            tx = rx - (center_x - x)
            ty = ry - (center_y - y)
            if np.sum(np.abs(base_color-img[y][x])) > 30 and (tx >= 0) and (tx < width) and (ty >= 0) and (ty < height):
                dst_img[ty][tx] = img[y][x]

#for i in range(200):
#    print img[0+i][401]
#    img[0+i][401] = np.array([0, 0, 255])
cv2.imwrite("result.png", dst_img)
#cv2.imshow("result", dst_img)
cv2.waitKey(0)
