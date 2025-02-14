# SIFT algorithm
import cv2

def sift_bf_simulate(
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

        # SIFT detector
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        kp1, des1 = sift.detectAndCompute(image, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # FLANN matcher using the metric L2 (Euclidean distance) to match descriptors
        # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # matches = matcher.knnMatch(des1, des2, 2)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

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

        # Pre-defined threshold for min. number of inliers to accept as a match: 0 (no boundaries are set to show other iterations with low matches)
        if len(good_matches) > 0:
            # These are the final lists of matches and matches data to be returned
            matches_info.append((
                # query_filename, test_filename, inliers_count, nfeatures, total_matches_without_filter, query_feature, test_feature, low_good_matches
                query_filename, 
                test_filename, 
                len(good_matches), 
                nfeatures, 
                len(matches),
                len(kp1), 
                len(kp2),
                len(good_matches)
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
    