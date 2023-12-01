import cv2
import matplotlib.pyplot as plt
import numpy as np

# Create a black image
image = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw a rectangle using OpenCV
cv2.rectangle(image, (50, 50), (250, 250), (255, 0, 0), 2)

# Display the image with Matplotlib
plt.imshow(image)
plt.title('OpenCV Drawing in Matplotlib')
plt.show()
