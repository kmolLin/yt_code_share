import cv2
import numpy as np
from numpy import array, uint8


def findcontour(img: np.ndarray):
    cv2.threshold(img, dst=img, *(89, 255, 0))

    cv2.morphologyEx(img, dst=img, *(2, array([[0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0, 0],
                                               [1, 1, 1, 1, 1, 1, 1],
                                               [0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0, 0]], dtype=uint8)), iterations=6)

    cv2.erode(img, dst=img, *(array([[0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [1, 1, 1, 1, 1],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0]], dtype=uint8),), iterations=1)

    cv2.dilate(img, dst=img, *(array([[0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0],
                                      [1, 1, 1, 1, 1, 1, 1],
                                      [0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0]], dtype=uint8),), **{'iterations': 8})
    return img


if __name__ == '__main__':

    img = cv2.imread("image1.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = img_gray.copy()
    process = findcontour(img_gray)
    process = cv2.Canny(process, 200, 255, apertureSize=5)
    if "3.0" < cv2.__version__ < "3.5":
        _, cnts, hierarchy = cv2.findContours(process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        cnts, hierarchy = cv2.findContours(process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)

    for c in cnts:
        if cv2.contourArea(c) < 300:
            continue
        M = cv2.moments(c)
        Cx = int(M["m10"] / M["m00"])
        Cy = int(M["m01"] / M["m00"])
        cv2.circle(img, (Cx, Cy), 10, (1, 227, 254), -1)

    cv2.imwrite("tt.jpg", img)
    cv2.namedWindow("image1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("compare", cv2.WINDOW_NORMAL)
    cv2.imshow("image1", img)
    cv2.imshow("compare", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
