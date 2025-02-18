# Import the OrFlannSac module
from fileio import load_images_from_folder
import match

'''
What do the function `load_images_from_folder` return?

load_images_from_folder('C:/Users/name/local_folder_path') 
    --> returns [[filename0, image0], [filename1, image1], ...]
'''
# Get the set of query and test images from the local folders "query" and "test"
query_imgs = load_images_from_folder('query')
test_imgs = load_images_from_folder('test')


'''
OPTION 1: Getting query_index by image file name
    - get the image file name and set it as the value for query_filename
    - the query_index will automatically get the image of the corresponding query_filename
'''
# query_filename = 'IMG_6845_jpeg.rf.ecd3c9b41e8cf9e0302536cbdbe1d11e.jpg'
# query_index = [query_imgs.index(i) for i in query_imgs if i[0] == query][0]
'''
OPTION 2: Getting query_index by the exact index of the image
    - set the index of the image you want as query image and set it to query_index variable
'''
query_index = 3


'''
query_filename: 
    - it gets the filename of the query image according to the set query_index
query_image:
    - it gets the IMAGE file of the query image according to the set query_index
'''
query_filename = query_imgs[query_index][0]
query_image = query_imgs[query_index][1]


'''
Start the algorithm:

AMADEO-BAUTISTA-LINGAD algorithm | (ABL) | Image Preprocessing + Bayesian Opt + ORB + FLANN + (MAGSAC++)

What do `matches` list variable return?
    - match_query_filename    => index [0]
    - match_test_filename     => index [1]
    - good_matches            => index [2]
    - nfeatures               => index [3]
    - total_matches           => index [4]

What do `visuals` list variable return?
    - None (its only process is to save the matched images from `matches`)
'''
matches, visuals = match.match( # initialize the algorithm to get the matches
    query_image=query_image, 
    query_filename=query_filename, 
    test_images=test_imgs
    ) 
match.save(visuals) # save the matched images to local folder "match_results"


# Check each value of the matches
for matched in matches:
    match_query_filename    = matched[0]
    match_test_filename     = matched[1]
    good_matches            = matched[2]
    nfeatures               = matched[3]
    total_matches           = matched[4]

    print("Total matches:\t\t", total_matches)
    print("Good matches:\t\t", good_matches)
    print("Nfeatures:\t\t", nfeatures)
    print("Query filename:\t\t", match_query_filename)
    print("Test filename:\t\t", match_test_filename)
