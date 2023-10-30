import cv2
import numpy as np

image = cv2.imread('Defect-image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_mask = cv2.inRange(gray, 128, 255)
white_mask = cv2.inRange(image, (255, 255, 255), (255, 255, 255))
image[np.where(gray_mask != 0)] = (0, 255, 0)
image[np.where(white_mask != 0)] = (0, 0, 255)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('output.jpg', image)
