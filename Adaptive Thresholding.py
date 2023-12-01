import cv2
import numpy as np

img = cv2.imread('sudoku.png')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, th1 = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('image', img)
# cv2.imshow('th1', th1)
cv2.imshow('th2', th2)
cv2.imshow('th3', th3)



cv2.waitKey(0)
cv2.destroyAllWindows()
