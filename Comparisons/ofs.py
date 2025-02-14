# OFS: ORB + FLANN + MAGSAC++ (ORFLANSAC)
import cv2
import numpy as np

def find_matches(
        query_image, 
        query_filename, 
        test_images, 
        nfeatures: int,
        estimator,
        preprocess_image,
        ):
    '''
    FUNCTION RETURNS:

    find_matches(query_image, query_filename, test_imgs, nfeatures)
        = 
        (
            [(query_filename, test_filename, inliers_count, nfeatures, total_matches_without_filter, query_feature, test_feature, first_good_matches), ...], 
            [(query_image_original, test_image_original, kp_query, kp_test, inlier_matches, query_filename, test_filename), ...]
        )

        IN SHORT
        = 
        (
            matches_info, 
            visualize_and_save_matches
        )
    '''

    # Save the original value of query_image
    query_image_original =  query_image

    # Array to store the final matches that will be RETURNED by this function
    matches_info = []

    # Image Pre-processing
    # query_image = preprocess_image(query_image)

    # ORB instantiation
    orb = cv2.ORB_create(nfeatures=int(nfeatures), scoreType=cv2.ORB_HARRIS_SCORE, fastThreshold=20, edgeThreshold=10)

    # Get keypoints and descriptors of Query Image
    kp_query, des_query = orb.detectAndCompute(query_image, None)

    print("LEN KP_Q:\t", len(kp_query))
    
    # FLANN matcher instantiation
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Array to save the comparison between query and test images
    visualize_and_save_matches = []
    
    # For each test image, compare the query image with it
    for test_filename, test_image in test_images:
        no_inliers = True

        # Save the original value of query_image
        test_image_original = test_image

        # Image prprocessing of test image
        # test_image = preprocess_image(test_image)
        
        # Get keypoints and descriptors of Test Image
        kp_test, des_test = orb.detectAndCompute(test_image, None)

        print("LEN KP_T:\t", len(kp_test))
        
        # Implement FLANN matcher
        # DATA: matches => ((query_match1, test_match2),(query_match2_1, test_match2_2), ...)
        matches = flann.knnMatch(des_query, des_test, k=2)

        print("SIMILAR MATCHES", len(matches))

        # Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2 and match[1].distance != 0:
                m, n = match
                if m.distance/n.distance < 0.75:    # if 0.75 (threshold) is to be changed, the closer to 0, the better and stricter.
                    good_matches.append(m)          # DATA: good_matches  => [desc_match1, desc_match2_1, ...]

        # If filtered matches are at least greater than X, then use MAGSAC++ to get inliers (or accurate matches)
        if len(good_matches) >= 0:
            num_total_matches = len(matches)
            first_good_matches = len(good_matches)
            num_good_matches = 0
            inlier_matches=[]

            # Convert keypoints to numpy arrays
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            try:
                # Find inliers using RANSAC (the model used is homography)
                _, mask = cv2.findHomography(src_pts, dst_pts, estimator, ransacReprojThreshold=10, maxIters=100)

                inliers = mask.ravel().tolist() # returns a 1D array of inliers with value 1 while the outliers have value of 0

                # Filter good matches based on inliers. Only keep good matches with x or more inliers
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]==True] # code is shortened through using list comprehensions
                num_good_matches = len(inlier_matches)
            except:
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                print("MAGSAC++ did not have enough number of filtered matches")
                print("Q filename: ", query_filename)
                print("src_pts: ", len(src_pts))
                print("T filename: ", test_filename)
                print("dst_pts: ", len(dst_pts))
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


                # If MAGSAC++ did not have enough number of filtered matches, then use Lowe's ratio test as good matches
                num_good_matches = len(good_matches)
                num_total_matches = len(matches)

            # Pre-defined threshold for min. number of inliers to accept as a match: x
            if len(inlier_matches) >= 0:

                # These are the final lists of matches data to be returned
                matches_info.append((
                    query_filename, 
                    test_filename, 
                    num_good_matches, 
                    nfeatures, 
                    num_total_matches,
                    len(kp_query),
                    len(kp_test),
                    first_good_matches
                    ))
                visualize_and_save_matches.append((
                    query_image_original, 
                    test_image_original, 
                    kp_query, kp_test, 
                    inlier_matches, 
                    test_filename,
                    query_filename
                    ))

    return (matches_info, visualize_and_save_matches)
