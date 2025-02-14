from optimizer import optimize_nfeatures
from ofs import find_matches
from fileio import visualize_and_save_matches
from preprocess import preprocess_image, preprocess_image_estacio_laurente, preprocess_none
import cv2

def preprocess_image_method(method_name='default'):
    match method_name:
        case "estacio_laurente":
            return preprocess_image_estacio_laurente
        case "none":
            return preprocess_none
        case "default":
            return preprocess_image

def match(
        query_image, 
        query_filename, 
        test_images, 
        nfeatures=1000,
        estimator=cv2.USAC_MAGSAC, 
        preprocess_img="default",
        ):
    '''
    FUNCTION RETURNS:

    match(query_image, query_filename, test_imgs, nfeatures)
        = 
        (
            [(query_filename, test_filename, inliers_count, nfeatures, total_matches_without_filter), ...], 
            [(query_image_original, test_image_original, kp_query, kp_test, inlier_matches, query_indicator, test_filename), ...]
        )
        
        IN SHORT
        = 
        (matches_info, visualize_and_save_matches)
    '''
    
    # Hyperparameter Optimization
    # nfeatures= 56946 # -->put the obtained average nfeatures value in this line
    nfeatures = optimize_nfeatures(query_image, query_filename, test_images, estimator, preprocess_image_method(preprocess_img))
    
    # Main modified ORB algorithm
    matches_info, visualize_and_save_matches = find_matches(query_image, query_filename, test_images, nfeatures, estimator, preprocess_image_method(preprocess_img))

    return (matches_info, visualize_and_save_matches)

# Function for saving the matched images to local folder
def save(visuals):
    for v in visuals:
        visualize_and_save_matches(v[0], v[1], v[2], v[3], v[4], v[5], v[6])