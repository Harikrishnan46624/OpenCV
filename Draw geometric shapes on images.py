import cv2

# img = cv2.imread('lena.jpg', 1)

import numpy as np

img = np.zeros([512, 512, 3], np.uint8)

# img = cv2.line(img, (0, 0), (255, 255), (0, 0, 255), 5) #Just a line

# img = cv2.arrowedLine(img, (0, 0), (255, 255), (255, 0, 0), 10) #arrowed line

# img = cv2.arrowedLine(img, (0, 255), (255, 255), (255, 0, 0), 10)


# img = cv2.rectangle(img, (384, 0), (510, 128), (0, 0, 255), 10) #to draw a rectangle

# img = cv2.rectangle(img, (384, 0), (510, 128), (0, 0, 255), -1) #to get filleed rectangle

# img = cv2.circle(img, (447, 63), 63, (0, 255, 0), -1)

font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
img = cv2.putText(img, 'OpenCV', (10, 500), font, 4, (0, 255, 255), 10, cv2.LINE_AA)







cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()