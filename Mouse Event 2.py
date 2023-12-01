import cv2
import numpy as np

def click_event(event, x, y, flags, param):
    # if event == cv2.EVENT_LBUTTONUP:
    #     cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    #     point.append((x, y))

    #     if len(point) >= 2:
    #         cv2.line(img, point[-1], point[-2], (255, 0, 0), 5)
    #     cv2.imshow('image', img)


    if event == cv2.EVENT_LBUTTONDOWN:
        blue = img[x, y, 0]
        green = img[x, y, 1]
        red = img[x, y, 2]
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        mycolorimage = np.zeros((512, 512, 3), np.uint8)
        
        mycolorimage[:] = [blue, green, red]
        cv2.imshow('color', mycolorimage)




img = cv2.imread('lena.jpg')
cv2.imshow('image', img)

point = []


cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()