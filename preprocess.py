"""
preprocess.py

This file contains the image preprocessing pipeline for the feature matching based object recognition system.

The purpose of this preprocessing is to prepare the images so that they are in a suitable form for feature detection and matching.

The preprocessing steps are as follows:

    1. Convert to Grayscale: This step is done to reduce the dimensionality of the image and to make the image processing faster.

    2. Apply Gaussian Blur: This step is done to reduce the noise in the image. Noise can lead to wrong feature detection and matching.

    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization): This step is done to enhance the contrast of the image. CLAHE is an adaptive histogram equalization technique that does not change the brightness of the image.

    4. Apply Canny Edge Detection: This step is done to detect the edges in the image. Canny edge detection is a popular edge detection algorithm that uses the gradient operator to detect edges.

The final preprocessed image is then used for feature detection and matching.

"""

import cv2

'''
    START: PREPROCESSING
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
    '''
        **PARAMETER CHANGES**
        
        ENHANCED
        cliplimit = 10.0
        tileGridSize = (4, 4)
        
        EXISTING CJER
        cliplimit = 4.0
        tileGridSize = (8, 8)
    '''
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    clahe_apply = clahe.apply(blurred_image)

    # Apply Canny Edge Detection (parameter adjusted to enhance contrast)
    edges = cv2.Canny(clahe_apply, threshold1=100, threshold2=100)

    combined_edges_clahe = cv2.addWeighted(clahe_apply, 0.8, edges, 0.1, 0.8)
    return combined_edges_clahe

def preprocess_image_estacio_laurente(image):
    # Convert to Grayscale (for dimensionality reduction)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_apply = clahe.apply(gray)

    return clahe_apply
