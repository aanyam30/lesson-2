import cv2
import numpy as np

image1 = cv2.imread('input1.jpg')
image2 = cv2.imread('input2.jpg')

sum = cv2.addWeighted(image1, 0.8, image2, 0.2, 0)
cv2.imshow('Adding images', sum)
cv2.waitKey(0)

diff = cv2.subtract(image1, image2)
cv2.imshow('Subtracting images', diff)
cv2.waitKey(0)

piku = cv2.imread('piku.png')
resize = cv2.resize(piku, (100, 100))
cv2.imshow('resized image', resize)
cv2.waitKey(0)

kernel = np.ones((5,5), np.uint8)
piku1 = cv2.erode(piku, kernel)
cv2.imshow('eroded image', piku1)
cv2.waitKey(0)

#gaussian blur is used mostly in machine learning pre processing steps
gaussian = cv2.GaussianBlur(piku, (7,7), 0)
cv2.imshow('blurred', gaussian)
cv2.waitKey(0)

median = cv2.medianBlur (piku, 9, 70)
cv2.imshow('blurred2', median)
cv2.waitKey(0)

bilateral = cv2.bilateralFilter (piku, 9, 70, 70)
cv2.imshow('blurred3', bilateral)
cv2.waitKey(0)

a = cv2.imread('building.jpg')
b = cv2.copyMakeBorder(a, 10, 10, 20, 20, cv2.BORDER_REFLECT, value=1000)
cv2.imshow('b', b)
cv2.waitKey(0)



cv2.destroyAllWindows() 