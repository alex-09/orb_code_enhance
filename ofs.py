# OFS: ORB + FLANN + MAGSAC++

import cv2
import numpy as np

'''
FUNCTION RETURNS:

find_matches(query_image, query_filename, test_imgs, nfeatures)
    = (
        [(query_filename, test_filename, inliers_count, nfeatures, total_matches_without_filter), ...], 
        [(query_image_original, test_image_original, kp_query, kp_test, inlier_matches, query_filename, test_filename), ...]
      )
    IN SHORT
    = (matches_info, visualize_and_save_matches)
'''
def find_matches(
        query_image, 
        query_filename, 
        test_images, 
        nfeatures,
        estimator,
        preprocess_image,
        filter_outlier=False
        ):
    query_image_original =  query_image

    # Array to store the final matches that will be RETURNED by this function
    matches_info = []

    if(preprocess_image == "estacio_laurente"):
        # resizing img1
        scale_percent = 100
        width = int(query_image.shape[1] * scale_percent / 100)
        height = int(query_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        query_image = cv2.resize(query_image, dim, interpolation = cv2.INTER_AREA)
    
    # OBJ 1 - Preprocess query image
    query_image = preprocess_image(query_image)

    # ORB instantiation
    orb = cv2.ORB_create(nfeatures=int(nfeatures), scoreType=cv2.ORB_HARRIS_SCORE, fastThreshold=20, edgeThreshold=10)
    kp_query, des_query = orb.detectAndCompute(query_image, None)
    
    # FLANN matcher instantiation
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Array to save the comparison between query and test images
    visualize_and_save_matches = []
    
    # For each test image, compare the query image with it
    for test_filename, test_image in test_images:
        test_image_original = test_image

        if(preprocess_image == "estacio_laurente"):
            #resizing img2
            scale_percent = 100
            width = int(test_image.shape[1] * scale_percent / 100)
            height = int(test_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            test_image = cv2.resize(test_image, dim, interpolation = cv2.INTER_AREA)
            test_image = cv2.resize(test_image, (query_image.shape[1], query_image.shape[0]))

        # OBJ 1 - Preprocess test image
        test_image = preprocess_image(test_image)
        
        kp_test, des_test = orb.detectAndCompute(test_image, None)
        matches = flann.knnMatch(des_query, des_test, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2 and match[1].distance != 0:
                m, n = match
                # print("=check if denom zero=============================")
                # print("m.distance: ", m.distance)
                # print("n.distance: ", n.distance)
                # print("------------------------------")
                # print("m.distance/n.distance: ", m.distance/n.distance)
                if m.distance/n.distance <= 0.75: # 25% closer points are considered as good matches
                    good_matches.append(m)

        # If filtered matches are at least greater than 10, then use MAGSAC++ to get inliers (or accurate matches)
        if len(good_matches) > 10:
            inlier_matches=[]

            if filter_outlier==False:
                inlier_matches = good_matches
            else:    
                # Convert keypoints to numpy arrays
                src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find inliers using RANSAC (the model used is homography)
                H, mask = cv2.findHomography(src_pts, dst_pts, estimator, ransacReprojThreshold=10, maxIters=100)
                inliers = mask.ravel().tolist() # returns a 1D array of inliers with value 1 while the outliers have value of 0

                # Filter good matches based on inliers. Only keep good matches with x or more inliers
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]==True] # code is shortened through using list comprehensions
            
            '''
            The longer version of the code for inlier_matches without using list comprehension:

            inlier_matches = []

            # Iterate over the items of good_matches
            for i in range(len(good_matches)):
                # Check if the current index is marked as an inlier
                if inliers[i] == True:
                    # If it is an inlier, append the corresponding match to the inlier_matches list
                    inlier_matches.append(good_matches[i])
            '''

            # Pre-defined threshold for min. number of inliers to accept as a match: 30
            if len(inlier_matches) > 10:
                # These are the final lists of matches and matches data to be returned
                matches_info.append((
                    query_filename, 
                    test_filename, 
                    len(inlier_matches), 
                    nfeatures, 
                    len(good_matches)
                    ))
                visualize_and_save_matches.append((
                    query_image_original, 
                    test_image_original, 
                    kp_query, kp_test, 
                    inlier_matches, 
                    query_filename, 
                    test_filename
                    ))

    return (matches_info, visualize_and_save_matches)
