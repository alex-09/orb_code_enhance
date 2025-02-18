import os
import cv2
import numpy as np

os.chdir('./AppFunctions')

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

visualize_and_save_matches(query_image, test_image, kp_query, kp_test, matches, test_filename)
    = None (its only function is to save the matched images)
'''
def visualize_and_save_matches(query_image, test_image, kp_query, kp_test, matches, test_filename):
    
    # Create a blank image to hold the combined output
    height, width = max(query_image.shape[0], test_image.shape[0]), query_image.shape[1] + test_image.shape[1]
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place the query image on the left
    combined_image[:query_image.shape[0], :query_image.shape[1]] = query_image
    
    # Place the test image on the right
    combined_image[:test_image.shape[0], query_image.shape[1]:] = test_image
    
    # Draw matches
    for match in matches:
        query_idx = match.queryIdx # Get the set of indexes of the query keypoint
        test_idx = match.trainIdx # Get the set of indexes of the test keypoint
        query_pt = kp_query[query_idx].pt # Get the coordinates of the query keypoint
        test_pt = kp_test[test_idx].pt # Get the coordinates of the test keypoint
        start_point = (int(query_pt[0]), int(query_pt[1]))
        end_point = (int(test_pt[0] + query_image.shape[1]), int(test_pt[1]))  # Shift x coordinate for test image
        cv2.line(combined_image, start_point, end_point, (0, 255, 0), 1)
        cv2.circle(combined_image, start_point, 5, (0, 0, 255), -1)
        cv2.circle(combined_image, end_point, 5, (0, 0, 255), -1)
    
    # Save the image comparison
    save_path = f"./match_results/matched_img_{test_filename}"
    res = cv2.imwrite(save_path, combined_image)
    # print(res, '\n', save_path)
