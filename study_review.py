# import cv2
# import numpy as np

# def nothing(x):
#     print(x)

# img = np.zeros((300, 512, 3), np.uint8)
# cv2.namedWindow('image')

# cv2.createTrackbar('B', 'image', 0, 255, nothing)
# cv2.createTrackbar('G', 'image', 0, 255, nothing)
# cv2.createTrackbar('R', 'image', 0, 255, nothing)
 
# while (1):
#     cv2.imshow('image', img)
#     k = cv2.waitKey(1) &0xFF 
#     if k == 27:
#         break
    

#     b = cv2.getTrackbarPos('B', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     r = cv2.getTrackbarPos('R', 'image')

#     img[:] = [b, g, r]

    

# cv2.destroyAllWindows()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = np.zeros([512, 512, 3], np.uint8)

# # img = cv2.line(img, (0, 0), (255, 255), (0, 0, 255), 5)

# # img = cv2.circle(img, (100, 100), 63, (0, 0, 255), 10)

# # img = cv2.rectangle(img, (30, 90), (120, 220), (255, 0, 0), 5)

# img = cv2.arrowedLine(img, (0, 0), (255, 255), (255, 0 ,0), 25)
# 
# img = cv2.ellipse(img, (200, 150), (100, 50), 30, 0, 360, (0, 225, 0), 2) 
# font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
# img = cv2.putText(img, 'HELLO',  (10, 500), font, 4, (0, 255, 255), 10)

# cv2.imshow('line', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
#     text = cv2.putText(frame, "HELLO", (10, 50), font, 1, (0,0,200), 5)
#     cv2.imshow('frame', text)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

    
# cap.release()
# cv2.destroyAllWindows()

# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

# img = np.zeros([512, 512, 0], np.uint8)
# import datetime
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:

#         font = cv2.FONT_HERSHEY_DUPLEX

#         text = str(datetime.datetime.now())
#         cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)
#         cv2.imshow("image", img)

# img = np.zeros([512, 512, 3], np.uint8)
# cv2.imshow('image', img)

# cv2.setMouseCallback('image', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = cv2.imread('messi5.jpg')

# b, g, r = cv2.split((img))

# height, width = img.shape[:2]

# print(f"Image Height: {height} pixels")
# print(f"Image Width: {width} pixels")

# img.resize((512, 512))

# print(img.shape)

# img = cv2.imread('gradient.png', 0)

# _, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
# _, th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# _, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# _, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# _, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

# imgage = [th1, th2, th3, th4, th5]
# for i in range(5):
#     plt.subplot(3, 2, i+1), plt.imshow(imgage[i])

# plt.show()


# img = cv2.imread('sudoku.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, th1 = cv2.threshold(img, 50,255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# image = [th1, th2, th3]

# for i in range(3):
#     plt.subplot(1, 3, i+1), plt.imshow(image[i])

# # plt.show()

# img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)

# _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# kernal = np.ones((5,5), np.uint8)

# dilation = cv2.dilate(mask, kernal, iterations=2)
# erosion = cv2.erode(mask, kernal, iterations=1)
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

# titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing']
# images = [img, mask, dilation, erosion, opening, closing]

# for i in range(6):
#     plt.subplot(3, 3, i+1), plt.imshow(images[i])
#     plt.title(titles[i])
# plt.show()

# img = cv2.imread('water.png', cv2.COLOR_BGR2RGB)
# img = cv2.imread('nahala.jpeg', cv2.COLOR_BGR2RGB)

# kernal = np.ones((5, 5), np.float32) / 25
# dst = cv2.filter2D(img, -1, kernal)
# blur = cv2.blur(img, (5, 5))
# gblur = cv2.GaussianBlur(img, (5, 5), 0)
# bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)
# median = cv2.medianBlur(img, 5)

# titles = ['image', '2D Convolution', 'blur', 'gblur', 'median', 'bilateralFilter']
# images = [img, dst, blur, gblur, median, bilateralFilter]

# for i in range(6):
#     plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])


# plt.show()


import cv2
import numpy as np

# Load the image
img = cv2.imread('lena.jpg')

# Function to create Gaussian pyramid
def build_gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

# Function to create Laplacian pyramid
def build_laplacian_pyramid(gaussian_pyramid):
    pyramid = [gaussian_pyramid[-1]]
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        expanded = cv2.pyrUp(gaussian_pyramid[i])
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], expanded)
        pyramid.insert(0, laplacian)
    return pyramid

# Number of pyramid levels
levels = 4

# Build Gaussian pyramid
gaussian_pyramid = build_gaussian_pyramid(img, levels)

# Build Laplacian pyramid
laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)

# Display the original image and pyramids
cv2.imshow('Original Image', img)

for i in range(levels):
    cv2.imshow(f'Gaussian Pyramid Level {i}', gaussian_pyramid[i])
    cv2.imshow(f'Laplacian Pyramid Level {i}', laplacian_pyramid[i])

cv2.waitKey(0)
cv2.destroyAllWindows()