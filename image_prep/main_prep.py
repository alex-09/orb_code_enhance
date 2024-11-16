# main aimge pre-processing

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# this image contains about 1MILLION pixels so expect high peaks in the histogram
image_location = 'query/unanghalik50_2_1600x.webp'

img1 = cv.imread(image_location, cv.IMREAD_GRAYSCALE)
assert img1 is not None, "file could not be read, check with os.path.exists()"

hist,bins = np.histogram(img1.flatten(),256,[0,256])
 
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
 
plt.plot(cdf_normalized, color = 'b')
plt.hist(img1.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()