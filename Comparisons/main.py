from fileio import load_images_from_folder, visualize_and_save_matches
import match          # AMA-BAU-LIN algorithm
from ex_orb import ex_orb   # EST-LAU algorithm
from orb import orb_simulate
from orb_bf import orb_bf_simulate
from sift import sift_simulate
from sift_bf import sift_bf_simulate
from akaze_bf import akaze_bf_simulate
from datetime import datetime
from create_csv import create_csv


NOW_PAST = datetime.now()

'''
What do they return?
-------------------------
load_images_from_folder('local_folder_path') 
    = [[filename0, image0], [filename1, image1], ...]
'''

# Load set of query and test images
query_imgs = load_images_from_folder('query')
test_imgs = load_images_from_folder('test')

avg_nfeatures = 0
iterations = 0

for img in query_imgs:
    matches = []
    visuals = []
    # REFERENCE:
    # query_image = img[1]
    # query_filename = img[0]

    # ALGO 1: AMA-BAU-LIN
    # matches, visuals = match.match(
    #     query_image=img[1], 
    #     query_filename=img[0], 
    #     test_images=test_imgs 
    # ) 

    # ALGO 2: EST-LAU
    # matches, visuals = ex_orb(
    #     query_image=img[1], 
    #     query_filename=img[0], 
    #     test_images=test_imgs,
    #     nfeatures=1000
    # )

    # ALGO 3: SIFT + FLANN
    # matches, visuals = sift_simulate(
    #     query_image=img[1], 
    #     query_filename=img[0], 
    #     test_images=test_imgs,
    #     nfeatures=0
    # )

    # ALGO 4: ORB + FLANN
    # matches, visuals = orb_simulate(
    #     query_image=img[1], 
    #     query_filename=img[0], 
    #     test_images=test_imgs,
    #     nfeatures=500
    # )

    # ALGO 5: SIFT + BF
    # matches, visuals = sift_bf_simulate(
    #     query_image=img[1], 
    #     query_filename=img[0], 
    #     test_images=test_imgs,
    #     nfeatures=0
    # )

    # ALGO 4: ORB + BF
    # matches, visuals = orb_bf_simulate(
    #     query_image=img[1], 
    #     query_filename=img[0], 
    #     test_images=test_imgs,
    #     nfeatures=500
    # )

    # ALGO 4: AKAZE + BF
    matches, visuals = akaze_bf_simulate(
        query_image=img[1], 
        query_filename=img[0], 
        test_images=test_imgs,
        nfeatures=500
    )

    create_csv(matches_info=matches)

    #############################################
    # Save the matched images to local folder
    match.save(visuals)

    print("LENGTH OF TOTAL MATCHES:", len(matches))

    for matched in matches:
        iterations += 1
        nfeatures = matched[3]
        total_matches = matched[4]
        good_matches = matched[2]
        query_features = matched[5]
        test_features = matched[6]
        lowe_good_matches = matched[7]

        # print("----------------------------------------------------------")
        # print("\nITERATIONS:\t", iterations)
        
        # # DEBUG INFO
        # print("Q IMAGE:\t\t", query_imgs.index(img)+1, " ", img[0])
        # print("INDEX TEST IMG:\t\t", matches.index(matched)+1)
        # print("Q filename:\t\t", matched[0])
        # print("T filename:\t\t", matched[1])
        # print("=======================================================")

        # print("Total matches:\t\t", total_matches)
        # print("Good matches:\t\t", good_matches)
        # print("Query Features:\t\t", query_features)
        # print("Test Features:\t\t", test_features)
        # print("Nfeatures:\t\t", nfeatures)
        # print("Lowe Good Matches:\t", lowe_good_matches)
        # print("----------------------------------------------------------")

        avg_nfeatures += nfeatures

avg_nfeatures = avg_nfeatures / 1500 # 1500 is the total number of iterations
print("\nAverage Nfeatures:\t", avg_nfeatures)

now = datetime.now()
print("Current time:", now)
print("Started time:", NOW_PAST)
