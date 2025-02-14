import cv2
import numpy as np
'''
AMA-BAU-LIN image preprocessing
combining these steps:
- grayscale
- gaussian blur
- clahe
- canny edge detection
'''
def preprocess_image(image):
    # Convert to Grayscale (for dimensionality reduction)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur (to reduce noise)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    clahe_apply = clahe.apply(blurred_image)
    # Apply Canny Edge Detection (parameter adjusted to enhance contrast)
    edges = cv2.Canny(clahe_apply, threshold1=100, threshold2=100)

    combined_edges_clahe = cv2.addWeighted(clahe_apply, 0.8, edges, 0.1, 0.8)
    return combined_edges_clahe

'''
EST-LAU image preprocessing
1. resize
2. clahe
'''
def preprocess_image_estacio_laurente(image):
    scale_percent = 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_apply = clahe.apply(image)

    return clahe_apply

'''
No image pre-processing.
We have to make the image grayscaled first before we can pass it to the ORB
'''
def preprocess_none(image):
    # Convert to Grayscale (for dimensionality reduction)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image



  

