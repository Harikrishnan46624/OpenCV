import cv2
import numpy as np

img1 = np.zeros((250, 500, 3), np.uint8)
img1 = cv2.rectangle(img1, (200, 0), (300, 100), (255, 255, 255), -1)

img2 = np.zeros((250, 500, 3), np.uint8)
# Set the left half to white
img2[:, :500 // 2] = [255, 255, 255]


bitand = cv2.bitwise_and(img2, img1)
bitor = cv2.bitwise_or(img2, img1)
bitxor = cv2.bitwise_xor(img1, img2)
bitnot1 = cv2.bitwise_not(img1)
bitnot2 = cv2.bitwise_not(img2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('bitAnd', bitor)


cv2.waitKey(0)
cv2.destroyAllWindows()