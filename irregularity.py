import numpy as np
import cv2
import math

img = cv2.imread("melanoma9.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret,thresh = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
im2, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = max(contours, key=cv2.contourArea)

(x,y), (Ma,ma), angle = cv2.fitEllipse(cnt)
(x,y) = (x/2,y/2)

area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, False)

I = (pow(perimeter,2)*Ma*ma) /( 2 *math.pi*(pow(Ma,2)+pow(ma,2))*area)
print(I)
