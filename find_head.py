import cv2

# Load the image
image = cv2.imread('nahala.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
face_detector = cv2.CascadeClassifier(r"E:\Programming\OpenCV\Nothing\haarcascade_frontalface_default.xml")
faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Iterate through detected faces
for face_coordinates in faces:
    x, y, w, h = face_coordinates
    
    # Draw a red square around the detected face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Show the image with the red squares around faces
cv2.imshow("Image with Red Squares", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
