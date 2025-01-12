# Estacio-Laurente ORB algorithm

import cv2 
import time
import numpy as np

def ex_orb(
        query_image, 
        query_filename, 
        test_images, 
        nfeatures
        ):
    # start time for time_complexity
    start_time = time.time()

    # Array to store the final matches that will be RETURNED by this function
    matches_info = []
    # Array to save the comparison between query and test images
    visualize_and_save_matches = []

    # image = cv2.imread("query/creamo_choco.webp", 2)
    # img2 = cv2.imread("test/test_creamo_mult.jpg", 2)

    image_orig = query_image
    image = cv2.imread(query_filename, 2)
    
    
    img2_orig = None

    for test_filename, img2 in test_images:
        img2_orig = img2
        img2 = cv2.imread("test/"+test_filename, 2)

        # resizing img1
        scale_percent = 100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        #resizing img2
        scale_percent = 100
        width = int(img2.shape[1] * scale_percent / 100)
        height = int(img2.shape[0] * scale_percent / 100)
        dim = (width, height)
        img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

        

        #enhance contrast using adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        img2 = clahe.apply(img2)

        #ORB detector
        orb = cv2.ORB_create(nfeatures, scoreType=1)
        kp1, des1 = orb.detectAndCompute(image, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
            key_size=12, multi_probe_level=1)
        search_params = dict(checks=50) # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        good_matches = []
        for index in range(len(matches)):
            if len(matches[index]) == 2:
                m, n = matches[index]
                if m.distance < 0.75 * n.distance: # threshold of ratio testing
                    matchesMask[index] = [1, 0]
                    good_matches.append(m)

        # function to prepare and draw the matches
        draw_params = dict( 
                        matchesMask = matchesMask,
                        flags = 2)
        match_result = cv2.drawMatchesKnn(image,kp1,img2,kp2,matches,None,**draw_params)

        # function to prepare and draw the keypoints
        keyp_draw_original = cv2.drawKeypoints(image, kp1, None)
        keyp_draw2 = cv2.drawKeypoints(img2, kp2, None)

        # Pre-defined threshold for min. number of inliers to accept as a match: 30
        if len(good_matches) > 10:
            # These are the final lists of matches and matches data to be returned
            matches_info.append((
                query_filename, 
                test_filename, 
                len(good_matches), 
                nfeatures, 
                len(matches)
                ))
            visualize_and_save_matches.append((
                image_orig, 
                img2_orig, 
                kp1, kp2, 
                good_matches, 
                query_filename, 
                test_filename
                ))
    
    # end time for time complexity
    end_time = time.time()
    sec_end_time = time.ctime(end_time)
    sec_start_time = time.ctime(start_time)

    # Total runtime of Estacio-Laurente ORB Algorithm
    total_runtime = f"Run-time: {end_time - start_time} secs"

    return (matches_info, visualize_and_save_matches)
    