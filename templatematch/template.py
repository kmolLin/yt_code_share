# template matching
import cv2
import numpy as np
from scipy import signal
from pprint import pprint
from matplotlib import pyplot as plt

def show_img(title, img):
    cv2.namedWindow(f"{title}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{title}", img)
    cv2.waitKey(0)


# 需要處理成透明背景 應該就可以找到對的物體進行辨識
path1 = "t2.jpg"
path2 = "template1.jpg"

img = cv2.imread(path1, 0)

template = cv2.imread(path2, 0)
w = template.shape[1]
h = template.shape[0]
mask_img = np.where(template > 0, 100, 0)
mask_img = np.float32(mask_img)

cv2.imwrite("mask.jpg", mask_img)

res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED, mask=mask_img)

cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)

loc = np.where(res > 0.85)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(len(loc[0]))
i = 0
tmp_point = []
x, y = (0, 0)
bottom_right = (max_loc[0] + w, max_loc[1] + h)
cv2.rectangle(img, max_loc, bottom_right, (0, 0, 0), 2)
imm = img[max_loc[1]:bottom_right[1], max_loc[0]:bottom_right[0]]
for pt in sorted(zip(*loc[::-1]), key=lambda s: s[0]):
    if np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2) < 500:
        continue

    x, y = pt
    tmp_point.append(pt)
    # print(pt, (pt[0] + w, pt[1] + h))
    # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)

x, y = (0, 0)
for pt in sorted(tmp_point, key=lambda s: s[1]):
    if np.sqrt((y - pt[1]) ** 2) < 150:
        continue
    x, y = pt
    print(pt)
    # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)
show_img("test", img)
cv2.imwrite("template.jpg", imm)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# print(min_val, max_val, min_loc, max_loc)
