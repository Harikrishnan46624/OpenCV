import cv2
import numpy as np
import matplotlib.pyplot as plt 

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Video", gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# import numpy as np

# img = np.zeros([512, 512, 3], np.uint8)

# img = cv2.line(img, (0, 0), (255, 255), (255, 255, 0), 4)

# img = cv2.circle(img, (447, 73), 63, (0, 255, 255), 6)

# img = cv2.rectangle(img, (0, 0), (200, 400), (0, 255, 255))

# cv2.imshow("IMage", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import datetime
# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text = str(datetime.datetime.now())
#     cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# def click_event(event, x, y, flag, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strXY = str(x) + ', ' + str(y)
#         cv2.putText(img, strXY, (x, y), font, 1, (0, 255, 255), 2, cv2.LINE_4)
#         cv2.imshow('image', img)

# img = cv2.imread('lena.jpg')
# cv2.imshow('image', img)
# cv2.setMouseCallback('image', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = cv2.imread('lena.jpg')
# print(img.shape)
# b, g, r = cv2.split((img))

# cv2.merge((b, g, r))

# cv2.resize(img, (600, 600))
# print(img.shape)

# cv2.imshow("im", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# img = np.zeros((300, 512, 3), np.uint8)
# def nothing(x):
#     pass

# cv2.namedWindow("Track Bar")

# cv2.createTrackbar('B', 'Track Bar', 0, 255, nothing)
# cv2.createTrackbar('G', 'Track Bar', 0, 255, nothing)
# cv2.createTrackbar('R', 'Track Bar', 0, 255, nothing)

# while True:
    
#     cv2.imshow('Track Bar', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     b = cv2.getTrackbarPos('B', 'Track Bar')
#     g = cv2.getTrackbarPos('G', 'Track Bar')
#     r = cv2.getTrackbarPos('R', 'Track Bar')

#     img[:] = [b, g, r]

# cv2.destroyAllWindows()


# img = cv2.imread('lena.jpg')

# def nothing(x):
#     pass

# cv2.namedWindow("img")

# cv2.createTrackbar('LH', 'img', 0, 255, nothing)
# cv2.createTrackbar('LS', 'img', 0, 255, nothing)
# cv2.createTrackbar('LV', 'img', 0, 255, nothing)
# cv2.createTrackbar("UH", "img", 255, 255, nothing)
# cv2.createTrackbar("US", "img", 255, 255, nothing)
# cv2.createTrackbar("UV", "img", 255, 255, nothing)

# while True:
#     img = cv2.imread('lena.jpg')
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     if cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE) <= 0:
#         break

#     l_h = cv2.getTrackbarPos('LH', 'img')
#     l_s = cv2.getTrackbarPos('LS', 'img')
#     l_v = cv2.getTrackbarPos('LV', 'img')

#     u_h = cv2.getTrackbarPos("UH", "img")
#     u_s = cv2.getTrackbarPos("US", "img")
#     u_v = cv2.getTrackbarPos("UV", "img")

#     l_b = np.array([l_h, l_s, l_v])
#     u_b = np.array([u_h, u_s, u_v])


#     mask = cv2.inRange(hsv, l_b, u_b)

#     res = cv2.bitwise_and(img, img, mask)

#     cv2.imshow('img', img)
#     cv2.imshow('Mask', mask)
#     cv2.imshow('result', res)


#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cv2.destroyAllWindows()

# img = cv2.imread('gradient.png', 0)
# _, th1 = cv2.threshold(img, 50, 225, cv2.THRESH_BINARY)
# _, th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# _, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# th = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# # cv2.imshow('image', img)
# # cv2.imshow('th1', th1)
# # cv2.imshow('th2', th2)
# # cv2.imshow('th3', th3)

# thresh = [th, th1, th2, th3]

# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(thresh[i])

# plt.show()


# # cv2.waitKey()
# # cv2.destroyAllWindows()


# img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
# _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# kernal = np.ones((5, 5), np.uint8)

# dilation = cv2.dilate(mask, kernal, iterations=2)
# erosion = cv2.erode(mask, kernal, iterations=1)
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

# morph = [dilation, erosion, opening, closing]

# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(morph[i])
# plt.show()

# img = cv2.imread('water.png', cv2.COLOR_BGR2RGB)

# kernal = np.ones((5, 5), np.float32) / 25

# dst = cv2.filter2D(img, -1, kernal)
# blur = cv2.blur(img, (5, 5))
# gblur = cv2.GaussianBlur(img, (5, 5), 0)
# median = cv2.medianBlur(img, 5)
# bilatrel = cv2.bilateralFilter(img, 9, 75, 75)

# images = [img, dst, blur, gblur, median, bilatrel]

# for i in range(6):
#     plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    
# plt.show()


# img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)

# lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# lap = np.uint8(np.absolute(lap))

# sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))

# sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# images = [img, lap, sobelX, sobelY, sobelCombined]
# for i in range(5):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')

# plt.show()

# img = cv2.imread("lena.jpg", 0)

# canny = cv2.Canny(img, 100, 200)
# images = [img, canny]

# for i in range(2):
#     plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
# plt.show()

img = cv2.imread("lena.jpg")

layer = img.copy()
gp = [layer]

for i in range(3):
    layer = cv2.pyrDown(layer)
    gp.append(layer)

layer = gp[2]
cv2.imshow("upper layer", layer)
lp = [layer]

cv2.waitKey()
cv2.destroyAllWindows()