# histogram equalization

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_location = 'query/unanghalik50_2_1600x.webp'

img = cv.imread(image_location, cv.IMREAD_GRAYSCALE)

equ = cv.equalizeHist(img)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
res = np.hstack((img,equ,cl1)) #stacking images side-by-side
cv.imwrite('result_histogram_eq.png',res)