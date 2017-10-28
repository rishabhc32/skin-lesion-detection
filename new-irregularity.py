import numpy as np
import cv2

img = cv2.imread("melanoma9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 32)
ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

im2, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

c = max(contours, key=cv2.contourArea)

rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)

extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

cv2.line(img,extLeft,extTop,(255,0,0),2)
cv2.line(img,extTop,extRight,(255,0,0),2)
cv2.line(img,extRight,extBot,(255,0,0),2)
cv2.line(img,extBot,extLeft,(255,0,0),2)

cv2.drawContours(img, c, -1, (0, 0, 255), 3)

cv2.imshow('frame', img)
cv2.waitKey(0)

