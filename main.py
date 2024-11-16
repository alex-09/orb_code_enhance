# Import the OrFlannSac module
from fileio import load_images_from_folder, visualize_and_save_matches
import match
from ex_orb import ex_orb
import window

'''
What do they return?
-------------------------
load_images_from_folder('local_folder_path') 
    = [[filename0, image0], [filename1, image1], ...]

optimize_nfeatures(query_image, query_filename, test_imgs)
    = (integer) --> optimal `nfeature`
'''

# Load set of query and test images
query_imgs = load_images_from_folder('query')
test_imgs = load_images_from_folder('test')

# PRINT QUERY IMGS index (for testing purposes, to know which image is being queried)
# for i in range(len(query_imgs)):
#     print(i, query_imgs[i][0])

# manually get a query image from a specific index
query_filename = query_imgs[0][0]
query_image = query_imgs[0][1]

# Start testing the algorithm for comparing the MODIFIED and EXSITING ORB algorithm
# Just uncomment the part with matches, visuals = ... code to start each test
# The scorings used for this simulation are 
#   Total matches,
#   Good matches,
#   Good match percentage
'''
TEST 1
AMADEO-BAUTISTA-LINGAD algorithm

(ABL) Image Preprocessing + Bayesian Opt + ORB + FLANN + (MAGSAC++)
'''
# matches, visuals = match.match(
#     query_image=query_image, 
#     query_filename=query_filename, 
#     test_images=test_imgs
#     ) 

'''
TEST 2
AMADEO-BAUTISTA-LINGAD algorithm with 
    IMG Preprocessing from ESTACTIO-LAURENTE

(EL) Img Preprocessing + Bayesian Opt + ORB + FLANN + (MAGSAC++)
'''
# matches, visuals = match.match(
#     query_image=query_image, 
#     query_filename=query_filename, 
#     test_images=test_imgs,
#     preprocess_img="estacio_laurente"
#     ) 

'''
TEST 3
AMADEO-BAUTISTA-LINGAD algorithm with 
    ESTACTIO-LAURENTE's
        IMG Preprocessing and
        FIXED NFEATURES of 1000

(EL) Img Preprocessing + (EL) Fixed Nfeatures (1000) + ORB + FLANN + (MAGSAC++) 
'''
matches, visuals = match.match(
    query_image=query_image, 
    query_filename=query_filename, 
    test_images=test_imgs,
    preprocess_img="estacio_laurente",
    fixed_nf=True,
    nfeatures=1000
    ) 

'''
TEST 4
AMADEO-BAUTISTA-LINGAD algorithm with 
    Fixed Nfeatures of 1000 from ESTACTIO-LAURENTE

(ABL) Image Preprocessing + (EL) Fixed Nfeatures (1000) + Bayesian Opt + ORB + FLANN + (MAGSAC++)
'''
# matches, visuals = match.match(
#     query_image=query_image, 
#     query_filename=query_filename, 
#     test_images=test_imgs,
#     fixed_nf=True,
#     nfeatures=1000
#     ) 

'''
TEST 5
ESTACIO-LAURENTE algorithm [Existing ORB Algorithm]

Fixed Nfeatures (1000) + CLAHE + ORB + FLANN 
'''
# matches, visuals = ex_orb(
# query_image=query_image, 
# query_filename=query_filename, 
# test_images=test_imgs,
# nfeatures=1000
# ) 

#############################################
# Save the matched images to local folder
match.save(visuals)

# print("Total number of matched images:\t", len(matches))

for matched in matches:
    total_matches = matched[4]
    good_matches = matched[2]
    good_match_percentage = (good_matches / total_matches) * 100

    # print("Verbose info:\t", match)
    print("(OBJ2) Total matches:\t\t", total_matches)                    # OBJ 2
    print("(OBJ1) Good matches:\t\t", good_matches)                      # OBJ 1
    print("(OBJ3) Good match pecentage:\t", good_match_percentage)       # OBJ 3
