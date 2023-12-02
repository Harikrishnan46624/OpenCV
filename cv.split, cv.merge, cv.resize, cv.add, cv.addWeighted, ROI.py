import cv2

img = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv-logo.png')

print("shape: ", img.shape)  #returns a tuple of number of rows, columns, and channels
print("size: ",img.size)  #returns Total number of pixels is accessed
print("dtype: ",img.dtype)  #returns Image datatype is obtained


b, g, r = cv2.split((img))  #s Image datatype is obtained
cv2.split(img) #output vector of arrays; the arrays themselves are reallocated, if needed.

img = cv2.merge((b, g, r))  #The number of channels will be the total number of channels in the matrix array.

# ROI (REGION OF INTERSET)

def find_cordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ' ,y)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', '+ str(y)
        cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        cv2.imshow('image', img)


ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

img = cv2.resize(img, (512, 512))
img2 = cv2.resize(img2, (512, 512))
# dst = cv2.add(img, img2)  # Calculates the per-element sum of two arrays or an array and a scalar.

dst = cv2.addWeighted(img, .9, img2, .1, 0);  #Calculates the weighted sum of two arrays


cv2.imshow('image', dst)

cv2.setMouseCallback('image', find_cordinates)


cv2.waitKey(0)
cv2.destroyAllWindows()







