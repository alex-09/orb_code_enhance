from optimizer import optimize_nfeatures
from ofs import find_matches
from fileio import visualize_and_save_matches
from preprocess import preprocess_image, preprocess_image_estacio_laurente
import cv2

def preprocess_image_method(method_name='default'):
    match method_name:
        case "estacio_laurente":
            return preprocess_image_estacio_laurente
        case "default":
            return preprocess_image

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
def match(
        query_image, 
        query_filename, 
        test_images, 
        estimator=cv2.USAC_MAGSAC, 
        fixed_nf=False,
        nfeatures=1000,
        preprocess_img="default",
        ):

    # Hyperparameter Optimization
    if fixed_nf == False:
        nfeatures = optimize_nfeatures(query_image, query_filename, test_images, preprocess_image_method(preprocess_img))
        # print("optimal nfeatures:", nfeatures)
    
    matches_info, visualize_and_save_matches = find_matches(query_image, query_filename, test_images, nfeatures, estimator, preprocess_image_method(preprocess_img))

    return (matches_info, visualize_and_save_matches)

# Function for saving the matched images to local folder
def save(visuals):
    for v in visuals:
        visualize_and_save_matches(v[0], v[1], v[2], v[3], v[4], v[5], v[6])