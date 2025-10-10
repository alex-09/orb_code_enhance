import os
import cv2
import numpy as np
import random

# os.chdir('./Comparisons')

def load_images_from_folder(folder):
    images = []
    try:
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append((filename, img))  # Store filename and image
    except FileNotFoundError:
        print("Fix the locating functions of the folders")
    return images

'''
FUNCTION RETURNS:

visualize_and_save_matches(query_image, test_image, kp_query, kp_test, matches, query_indicator, test_filename)
    = None (its only function is to save the matched images)
'''
def visualize_and_save_matches(query_image, test_image, kp_query, kp_test, matches, test_filename, query_filename, bad_matches=[], total_matches=1):

    sampled_bad_matches = random.sample(bad_matches, 100)

    # --- 1. Define Label Height ---
    LABEL_HEIGHT = 40  # Height of the white label bar in pixels
    
    # Create a blank image to hold the combined output
    # height, width = max(query_image.shape[0], test_image.shape[0]), query_image.shape[1] + test_image.shape[1]

    original_height = max(query_image.shape[0], test_image.shape[0])
    width = query_image.shape[1] + test_image.shape[1]
    
    height = original_height + LABEL_HEIGHT

    combined_image = np.zeros((height, width, 3), dtype=np.uint8)

    # --- 2. Create White Background for Label ---
    # Fill the entire image with white first (255, 255, 255)
    combined_image.fill(255)
    
    # TODO - query img is (640,640) and test img is (640,640,3)
    # Place the query image on the left
    combined_image[:query_image.shape[0], :query_image.shape[1]] = query_image
    
    # Place the test image on the right
    combined_image[:test_image.shape[0], query_image.shape[1]:] = test_image
    
    for match in sampled_bad_matches:
        query_idx = match.queryIdx
        test_idx = match.trainIdx
        query_pt = kp_query[query_idx].pt
        test_pt = kp_test[test_idx].pt
        
        start_point = (int(query_pt[0]), int(query_pt[1]))
        end_point = (int(test_pt[0] + query_image.shape[1]), int(test_pt[1])) 

        # Draw circles in pink
        cv2.circle(combined_image, start_point, 4, (255, 0, 255), -1) 
        cv2.circle(combined_image, end_point, 4, (255, 0, 255), -1) 

    # Draw matches
    for match in matches:
        # print(match)
        query_idx = match.queryIdx # Get the set of indexes of the query keypoint
        test_idx = match.trainIdx # Get the set of indexes of the test keypoint
        query_pt = kp_query[query_idx].pt # Get the coordinates of the query keypoint
        test_pt = kp_test[test_idx].pt # Get the coordinates of the test keypoint
        start_point = (int(query_pt[0]), int(query_pt[1]))
        end_point = (int(test_pt[0] + query_image.shape[1]), int(test_pt[1]))  # Shift x coordinate for test image
        cv2.line(combined_image, start_point, end_point, (0, 255, 0), 1)

        # inlier colors - cyan
        cv2.circle(combined_image, start_point, 5, (0, 255, 0), -1)
        cv2.circle(combined_image, end_point, 5, (0, 255, 0), -1)

        # outlier colors - red
        # cv2.circle(combined_image, start_point, 3, (0, 0, 255), -1)
        # cv2.circle(combined_image, end_point, 3, (0, 0, 255), -1)

    # --- 4. Add Text Label to the White Area ---
    label_text = f"inliers: {len(matches)} | outliers: {len(bad_matches)} | match percentage: {round((len(matches)/total_matches)*100, 2)}% | total matches: {total_matches}"
    
    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 0, 0) # Black text (BGR)
    thickness = 2
    
    # Calculate text size for centering
    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
    
    # Calculate text position (Centered horizontally, near the top of the label area)
    text_x = (width - text_size[0]) // 2
    # Place text in the white area (original_height is the y-start of the white bar)
    text_y = original_height + (LABEL_HEIGHT + text_size[1]) // 2 
    
    cv2.putText(combined_image, label_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    
    # Save the image comparison
    save_path = f"results/{query_filename}_matched_img_{test_filename}"
    # save_path = f"results/obj 2"
    res = cv2.imwrite(save_path, combined_image)
