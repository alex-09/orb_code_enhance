# Estacio-Laurente ORB algorithm
import cv2
import numpy as np

def ex_orb(
        query_image, 
        query_filename, 
        test_images, 
        nfeatures
        ):

    # Array to store the final matches that will be RETURNED by this function
    matches_info = []
    # Array to save the comparison between query and test images
    visualize_and_save_matches = []

    # image = cv2.imread("query/creamo_choco.webp", 2)
    # img2 = cv2.imread("test/test_creamo_mult.jpg", 2)

    image_orig = query_image
    image = cv2.imread("query/"+query_filename, 2)
    
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
        if len(good_matches) >= 0:
            num_total_matches = len(matches)
            first_good_matches = len(good_matches) # from ratio test
            num_good_matches = 0
            inlier_matches=[]

            # Convert keypoints to numpy arrays
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            try:
                # Find inliers using RANSAC (the model used is homography)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold=10, maxIters=100)

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
            
            # These are the final lists of matches and matches data to be returned
            matches_info.append((
                # query_filename, test_filename, inliers_count, nfeatures, total_matches_without_filter, query_feature, test_feature, low_good_matches
                query_filename, 
                test_filename, 
                num_good_matches, 
                nfeatures, 
                num_total_matches,
                len(kp1), 
                len(kp2),
                first_good_matches
                ))
            visualize_and_save_matches.append((
                image_orig, 
                img2_orig, 
                kp1, kp2, 
                good_matches, 
                query_filename, 
                test_filename
                ))

    return (matches_info, visualize_and_save_matches)
    