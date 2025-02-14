# OFS: ORB + FLANN + MAGSAC++
# For testing purposes

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
    = (
        matches_info, 
        visualize_and_save_matches
      )
'''
def find_matches(
        query_image, 
        query_filename, 
        test_images, 
        nfeatures,
        estimator,
        preprocess_image,
        filter_outlier,
        fixed_nf,
        algo_name="default"
        ):
    
    #synthetic noisezz
    # Generate Gaussian noise
    # noise = np.random.normal(0.001, 1, query_image.shape).astype(np.uint8)
    # # Add the noise to the original image
    # query_image = cv2.add(query_image, noise)
    # query_image= cv2.GaussianBlur(query_image, (5, 5), 0)
    # beta = max(-255, min(255, -5))
    # # Adjust brightness
    # query_image = cv2.convertScaleAbs(query_image, alpha=1, beta=beta)

    query_image_original =  query_image

    # Array to store the final matches that will be RETURNED by this function
    matches_info = []

    # Sturcture is for testing: Image preprocessing
    #   We separated the initiation of image preprocessing 
    #   because of different image encoding.
    #   For Est-Lau ORB they used imread(, 2) to grayscale the image and 
    #   for ORFLANSAC we used imread(, 0) then convert to BGR2GRAY  to grayscale the image
    if algo_name=="ex_orb":
        query_image = cv2.imread("query/"+query_filename, 2)
        query_image = preprocess_image(query_image)
    elif algo_name=="default":
        # OBJ 1 - Preprocess query image
        query_image = preprocess_image(query_image)

    # ORB instantiation
    # The set ORB parameters used in ORFLANSAC are different from the ones used in Est-Lau ORB
    orb = None
    if algo_name=="default": 
        #indication that it is for ORFLANSAC
        orb = cv2.ORB_create(nfeatures=int(nfeatures), scoreType=cv2.ORB_HARRIS_SCORE, fastThreshold=20, edgeThreshold=10)
    elif algo_name=="ex_orb": 
        #indication that it is for Est-Lau ORB
        orb = cv2.ORB_create(nfeatures=int(nfeatures), scoreType=cv2.ORB_FAST_SCORE)

    kp_query, des_query = orb.detectAndCompute(query_image, None)
    
    # FLANN matcher instantiation
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Array to save the comparison between query and test images
    visualize_and_save_matches = []
    
    # For each test image, compare the query image with it
    for test_filename, test_image in test_images:
        test_image_original = test_image

        # Testing: Image prprocessing of test image
        if algo_name=="ex_orb":
            test_image = cv2.imread("test/"+test_filename, 2)
            test_image = preprocess_image(test_image)
        elif algo_name=="default":
            # OBJ 1 - Preprocess test image
            test_image = preprocess_image(test_image)
        
        kp_test, des_test = orb.detectAndCompute(test_image, None)
        matches = flann.knnMatch(des_query, des_test, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

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
                if m.distance/n.distance < 0.75: # 25% closer points are considered as good matches
                    good_matches.append(m)

        # If filtered matches are at least greater than 10, then use MAGSAC++ to get inliers (or accurate matches)
        if len(good_matches) > 10:
            inlier_matches=[]
            outlier_matches=[]
            
            # Their data is
            # matches = ((match1,match2),(match2_1, match2_2), ...)
            # good_matches = [match1, match2_1, ...]
            # 
            # matches_list = []
            # for i in list(matc):
            #     for j in i:
            #         matches_list.append(j)


            if filter_outlier==False:
                inlier_matches = good_matches
                good_matches = matches
            else:    
                # Convert keypoints to numpy arrays
                src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find inliers using RANSAC (the model used is homography)
                _, mask = cv2.findHomography(src_pts, dst_pts, estimator, ransacReprojThreshold=10, maxIters=100)
                inliers = mask.ravel().tolist() # returns a 1D array of inliers with value 1 while the outliers have value of 0

                # Filter good matches based on inliers. Only keep good matches with x or more inliers
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]==True] # code is shortened through using list comprehensions
            
                outlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]==False]


                # START: FOR TESTING: This section if for visualizing the inliers and outliers 
                # Create a new image to draw matches
                # draw_img = np.hstack((query_image_original, test_image_original))  # Combine images side by side

                # # Iterate over matches to draw inliers and outliers
                # for i in range(len(good_matches)):
                #     pt1 = tuple(np.round(kp_query[good_matches[i].queryIdx].pt).astype(int))
                #     pt2 = tuple(np.round(kp_test[good_matches[i].trainIdx].pt).astype(int))

                #     # Adjust point positions for combined image
                #     pt2_adjusted = (pt2[0] + query_image_original.shape[1], pt2[1])  # Shift pt2 to the right

                #     if mask[i]:  # Inlier
                #         color = (0, 255, 0)  # green for inliers
                        
                #         # Draw line for inliers
                #         color_line = (100, 0, 100)
                #         cv2.line(draw_img, pt1, pt2_adjusted, color_line, 1)

                #         # Draw points for inliers
                #         cv2.circle(draw_img, pt1, 3, color, -1)     # Point in img1
                #         cv2.circle(draw_img, pt2_adjusted, 3, color, -1)  # Point in img2
                #     else:  # Outlier
                #         color = (0, 0, 255)  # Red for outliers
                #         # Draw points for outliers
                #         cv2.circle(draw_img, pt1, 3, color, -1)     # Point in img1
                #         cv2.circle(draw_img, pt2_adjusted, 3, color, -1)  # Point in img2

                # test_window_title = f"[Nfeatures = {str(nfeatures)}] [Inlier Matches = {str(len(inlier_matches))}] [Outliers = {str(len(outlier_matches))}]"
                # cv2.imshow(test_window_title, draw_img)
                # cv2.waitKey(0)
                # END: FOR TESTING: This section if for visualizing the inliers and outliers 

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

            # Pre-defined threshold for min. number of inliers to accept as a match: x
            if len(inlier_matches):
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
                    query_filename, # query_indicator
                    test_filename
                    ))

    return (matches_info, visualize_and_save_matches)
