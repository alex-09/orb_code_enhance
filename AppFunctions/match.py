from optimizer import optimize_nfeatures
import amadeo_bautista_lingad
from fileio import visualize_and_save_matches
from preprocess import preprocess_image
import cv2


def match(
        query_image, 
        query_filename, 
        test_images, 
        nfeatures=1000,
        estimator=cv2.USAC_MAGSAC, 
        ):
    '''
    FUNCTION RETURNS:

    match(query_image, query_filename, test_imgs, nfeatures)
        = (
            [(query_filename, test_filename, inliers_count, nfeatures, total_matches_without_filter), ...], 
            [(query_image_original, test_image_original, kp_query, kp_test, inlier_matches, query_indicator, test_filename), ...]
        )
        IN SHORT
        = (matches_info, visualize_and_save_matches)
    '''

    # Hyperparameter Optimization
    nfeatures = optimize_nfeatures(query_image, query_filename, test_images, estimator, preprocess_image)
    
    # Find matches using AMADEO-BAUTISTA-LINGAD ORB algorithm
    matches_info, visualize_and_save_matches = amadeo_bautista_lingad.find_matches(query_image, query_filename, test_images, nfeatures, estimator, preprocess_image)
    
    return (matches_info, visualize_and_save_matches)

# Function for saving the matched images to local folder
def save(visuals):
    for v in visuals:
        visualize_and_save_matches(v[0], v[1], v[2], v[3], v[4], v[5])