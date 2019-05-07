import cv2
import math
import numpy as np
import os

# 读取图片
img = cv2.imread("C:\\Users\\ThinkPad\\Desktop\\test\\19.jpg")
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 显示消除椒盐噪声（采用中值滤波器-相对均值滤波器效果会好一点） 或者采用高斯去噪
median = cv2.medianBlur(img, 5)
cv2.imshow("Median", median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 将图片数据类型转换为灰度图
img_gray = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图片边缘检测再膨胀
edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
img_open = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
cv2.imshow("now", img_open)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("C:\\Users\\ThinkPad\\Desktop\\test\\20.jpg", img_open)
