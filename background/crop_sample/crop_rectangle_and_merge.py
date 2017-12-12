import sys

from PIL import Image, ImageOps
import cv2
import numpy as np

division_num = 3
args = sys.argv

img = cv2.imread(args[1])
divided_img = [[None for j in range(division_num)] for i in range(division_num)]

detector = cv2.FeatureDetector_create("FAST")

height, width, channels = img.shape
divided_width = width/division_num
divided_height = height/division_num
dst_img = np.zeros((height, width, 3), np.uint8)

complement_index = []
keypoints_order = []

for y in range(division_num): # height
    for x in range(division_num): # width
        tmp_img = img[y*divided_height:(y+1)*divided_height, x*divided_width:(x+1)*divided_width]
        #if y == 0 and x == 1:
        #    cv2.imshow("hoge", tmp_img)
        #    cv2.waitKey(0)
        tmp_keypoints = detector.detect(tmp_img)
        tmp_keypoints_pt = np.array([[tmp_keypoints[i].pt] for i in range(len(tmp_keypoints))]).astype(np.int32)
        keypoints_order.append((len(tmp_keypoints_pt), (y, x)))
        if len(tmp_keypoints_pt) < 3:
            complement_index.append((y, x))
            continue
        tx,ty,tw,th = cv2.boundingRect(tmp_keypoints_pt)
        if tx < 10:
            tw += tx
            tx = 0
        if ty < 10:
            th += ty
            ty = 0
        if tx+tw > divided_width-1-10:
            tw = divided_width-tx
        if ty+th > divided_height-1-10:
            th = divided_height-ty
        tmp_img = tmp_img[ty:ty+th, tx:tx+tw]
        tmp_img_flipx = cv2.flip(tmp_img, 1)
        tmp_img_flipy = cv2.flip(tmp_img, 0)
        tmp_img_flipxy = cv2.flip(tmp_img_flipx, 0)
        #print(tx, ty, tw, th)
        for sy in range(divided_height):
            for sx in range(divided_width):
                flipy = sy/th
                flipx = sx/tw
                if flipy%2 == 0:
                    if flipx%2 == 0:
                        dst_img[y*divided_height+sy, x*divided_width+sx][:] = tmp_img[sy%th, sx%tw]
                    else:
                        dst_img[y*divided_height+sy, x*divided_width+sx][:] = tmp_img_flipx[sy%th, sx%tw]
                else:
                    if flipx%2 == 0:
                        dst_img[y*divided_height+sy, x*divided_width+sx][:] = tmp_img_flipy[sy%th, sx%tw]
                    else:
                        dst_img[y*divided_height+sy, x*divided_width+sx][:] = tmp_img_flipxy[sy%th, sx%tw]
        cv2.imshow("result", dst_img)
        cv2.waitKey(0)

keypoints_order.sort()
for cy, cx in complement_index:
    good = keypoints_order.pop(len(keypoints_order)-1)
    y, x = good[1]
    tmp_img = img[y*divided_height:(y+1)*divided_height, x*divided_width:(x+1)*divided_width]
    tmp_keypoints = detector.detect(tmp_img)
    tmp_keypoints_pt = np.array([[tmp_keypoints[i].pt] for i in range(len(tmp_keypoints))]).astype(np.int32)
    tx,ty,tw,th = cv2.boundingRect(tmp_keypoints_pt)
    if tx < 10:
        tw += tx
        tx = 0
    if ty < 10:
        th += ty
        ty = 0
    if tx+tw > divided_width-1-10:
        tw = divided_width-tx
    if ty+th > divided_height-1-10:
        th = divided_height-ty
    tmp_img = tmp_img[ty:ty+th, tx:tx+tw]
    tmp_img_flipx = cv2.flip(tmp_img, 1)
    tmp_img_flipy = cv2.flip(tmp_img, 0)
    tmp_img_flipxy = cv2.flip(tmp_img_flipx, 0)
    for sy in range(divided_height):
        for sx in range(divided_width):
            flipy = sy/th
            flipx = sx/tw
            if flipy%2 == 0:
                if flipx%2 == 0:
                    dst_img[cy*divided_height+sy, cx*divided_width+sx][:] = tmp_img[sy%th, sx%tw]
                else:
                    dst_img[cy*divided_height+sy, cx*divided_width+sx][:] = tmp_img_flipx[sy%th, sx%tw]
            else:
                if flipx%2 == 0:
                    dst_img[cy*divided_height+sy, cx*divided_width+sx][:] = tmp_img_flipy[sy%th, sx%tw]
                else:
                    dst_img[cy*divided_height+sy, cx*divided_width+sx][:] = tmp_img_flipxy[sy%th, sx%tw]

cv2.imshow("result", dst_img)
cv2.waitKey(0)


keypoints = detector.detect(img)
keypoints_pt = np.array([[keypoints[i].pt] for i in range(len(keypoints))]).astype(np.int32)

hull = cv2.convexHull(keypoints_pt, returnPoints=False)
#print([np.array([keypoints_pt[p[0]] for p in hull])])
out = cv2.drawKeypoints(img, [keypoints[p[0]] for p in hull], None)
#out = cv2.drawKeypoints(img, keypoints, None)
cv2.polylines(img, [np.array([keypoints_pt[p[0]] for p in hull])], True, (0, 255, 255))

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.imwrite(args[1]+"_result.jpg", dst_img)
#cv2.imshow("result", dst_img)
#cv2.waitKey(0)
