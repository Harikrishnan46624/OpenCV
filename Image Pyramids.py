# GAUSSIAN PYRAMID

# import cv2
# import numpy as np

# img = cv2.imread('nahala.jpeg')

# lr1 = cv2.pyrDown(img)
# lr2 = cv2.pyrDown(lr1)

# hr1 = cv2.pyrUp(lr2)

# cv2.imshow('image', img)
# cv2.imshow('lr1', lr1)
# cv2.imshow('lr2', lr2)
# cv2.imshow('hr1', hr1)


# cv2.waitKey()
# cv2.destroyAllWindows()



#MULTIPLE IMAGE PYRAMID



import cv2
import numpy as np

img = cv2.imread('nahala.jpeg')

layer = img.copy()
gp = [layer]

# Generate Gaussian pyramid
for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)

layer = gp[5]
cv2.imshow("upper layer", layer)
lp = [layer]

# Generate Laplacian pyramid
for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gp[i])
    
    # Ensure the size of the images match
    h, w, _ = gp[i-1].shape
    gaussian_extended = cv2.resize(gaussian_extended, (w, h))

    laplacian = cv2.subtract(gp[i-1], gaussian_extended)
    cv2.imshow(str(i), laplacian)

cv2.imshow('image', img)

cv2.waitKey()
cv2.destroyAllWindows()
