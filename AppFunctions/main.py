# Import the OrFlannSac module
from fileio import load_images_from_folder
import match

'''
What do they return?
-------------------------
load_images_from_folder('local_folder_path') 
    = [[filename0, image0], [filename1, image1], ...]
'''

# Load set of query and test images
query_imgs = load_images_from_folder('query')
test_imgs = load_images_from_folder('test')

# manually get the query image file name and set it to query to set the query image
# query = 'IMG_6845_jpeg.rf.ecd3c9b41e8cf9e0302536cbdbe1d11e.jpg'
# query_index = [query_imgs.index(i) for i in query_imgs if i[0] == query][0]
query_index = 3
query_filename = query_imgs[query_index][0]
query_image = query_imgs[query_index][1]

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
matches, visuals = match.match(
    query_image=query_image, 
    query_filename=query_filename, 
    test_images=test_imgs
    ) 


#############################################
# Save the matched images to local folder
match.save(visuals)

for matched in matches:
    total_matches = matched[4]
    good_matches = matched[2]
    good_match_percentage = (good_matches / total_matches) * 100

    # print("Verbose info:\t", match)
    print("Total matches:\t\t", total_matches)                    # OBJ 2
    print("Good matches:\t\t", good_matches)                      # OBJ 1
    print("Good match pecentage:\t", good_match_percentage)       # OBJ 3
