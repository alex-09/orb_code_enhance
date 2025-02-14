import csv
import os


def create_csv(matches_info, filename='file_ABL_no_img_enhancement.csv'):
    '''
    PARAMS: matches_info, filename.csv (or no value for default)
    '''
    file_exists = os.path.isfile(filename)

    try:
        
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(['query_filename', 'test_filename', 'num_good_matches', 
                                 'nfeatures', 'num_total_matches', 'query_features', 
                                 'test_features', 'lowe_good_matches'])

            for info in matches_info:
                writer.writerow(info)

        print(f"Data written to {filename}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")