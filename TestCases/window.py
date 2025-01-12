import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QCoreApplication, Qt
from ex_orb import ex_orb
from fileio import load_images_from_folder
import match as mod_orb
import numpy as np

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Upload and Display")
        self.setGeometry(100, 100, 800, 600)

        # Layout
        self.layout = QVBoxLayout()

        # Algo Label
        self.algo_label = QLabel("No Algorithm executed yet")
        self.layout.addWidget(self.algo_label)

        # Algo button
        self.algo_button = QPushButton("Est-Lau ORB")
        self.algo_button.clicked.connect(self.run_ex_orb)
        self.layout.addWidget(self.algo_button)

        # Algo button - Mod ORB
        self.algo_mod_orb_btn = QPushButton("Mod-ORB")
        self.algo_mod_orb_btn.clicked.connect(self.run_mod_orb)
        self.layout.addWidget(self.algo_mod_orb_btn)
        
        # Image display label
        self.layout_image = QHBoxLayout()

        self.image_label = QLabel("Upload an image")
        self.image_label_matched = QLabel("")
        self.layout_image.addWidget(self.image_label)
        self.layout_image.addWidget(self.image_label_matched)

        self.layout.addLayout(self.layout_image)

        # Section for result of image
        self.layout_result_img = QVBoxLayout()

        self.image_label_result = QLabel("RESULT IMAGE")
        self.layout_result_img.addWidget(self.image_label_result)
        self.layout.addLayout(self.layout_result_img)   

        # Display bottom buttons
        self.layout_bottom_btns = QHBoxLayout()
        
        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.layout_bottom_btns.addWidget(self.upload_button)

        # Delete button
        self.delete_button = QPushButton("Delete Image")
        self.delete_button.clicked.connect(self.delete_image)
        self.layout_bottom_btns.addWidget(self.delete_button)

        # Save results button
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_results)
        self.layout_bottom_btns.addWidget(self.save_results_button)

        self.layout.addLayout(self.layout_bottom_btns)

        # Set layout to the window
        self.setLayout(self.layout)

        # Initialize image path
        self.image_path = None

        # Visuals container
        self.visual_container = None


    def upload_image(self):
        # Open file dialog to select an image
        file_name = QFileDialog.getOpenFileName(self, self.tr("Open Image"), "C:\\Users\\johnl\\Codes\\ORBv2\\query", self.tr("Image Files (*.png *.jpg *.bmp *.webp)"))

        # Print the path of the selected image
        print(file_name[0], type(file_name))

        if file_name:
            self.image_path = file_name[0]
            self.image_label.clear()
            self.image_label_matched.clear()

            self.display_image()
            
        else:
            print("No image selected")

    def display_image(self):
        self.image_label.clear()
        self.image_label_matched.clear()

        if self.image_path:
            # Read and convert the image using OpenCV
            image = cv2.imread(self.image_path)
            if image is not None:
                # Convert BGR to RGB format for displaying in PyQt
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channel = image.shape
                bytes_per_line = channel * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Set the pixmap to the label for display
                self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(700, 300, Qt.AspectRatioMode.KeepAspectRatio))
                print(self.image_label.size())

    def delete_image(self):
        # Clear the displayed image and reset image path
        self.image_label.clear()
        self.image_path = None

    def run_ex_orb(self):
        self.algo_label.setText("Running ORB")

        '''
        TEST 5
        ESTACTIO-LAURENTE algorithm [Existing ORB Algorithm]

        Fixed Nfeatures (1000) + CLAHE + ORB + FLANN 
        '''
        image_path = self.image_path
        query_image = cv2.imread(image_path)
        query_filename = image_path
        test_imgs = load_images_from_folder(r"C:\Users\johnl\Codes\ORBv2\test")
        self.results = r"C:\Users\johnl\Codes\ORBv2\results"

        print("Image path:", image_path)
        print("Query filename:", query_filename)

        matches, visuals = ex_orb(
            query_image=query_image, 
            query_filename=query_filename, 
            test_images=test_imgs,
            nfeatures=1000
            ) 
        
        self.visual_container = visuals
        
        for match in matches:
            total_matches = match[4]
            good_matches = match[2]
            good_match_percentage = (good_matches / total_matches) * 100

            # print("Verbose info:\t", match)
            print("\nESTACIO-LAURENTE ORB algorithm")
            print("(OBJ2) Total matches:\t\t", total_matches)                    # OBJ 2
            print("(OBJ1) Good matches:\t\t", good_matches)                      # OBJ 1
            print("(OBJ3) Good match pecentage:\t", good_match_percentage)       # OBJ 3
            print("+++++++++++++++")
            print()

        # Convert BGR to RGB format for displaying in PyQt
        image = cv2.cvtColor(visuals[0][1], cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = channel * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # self.image_label_matched.setPixmap(QPixmap.fromImage(q_image).scaled(700, 300, Qt.AspectRatioMode.KeepAspectRatio))

        # concatenate an image below the self.image_label_matched with another image
        mod_orb.save(visuals)
        q_imgs= []
        for v in visuals:
            # Read and convert the image using OpenCV
            image = cv2.imread(f"C:/Users/johnl/Codes/ORBv2/results/matched_img_{v[6]}")
            if image is not None:
                # Convert BGR to RGB format for displaying in PyQt
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channel = image.shape
                bytes_per_line = channel * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                q_imgs.append(q_image)

        qq = None
        for q in q_imgs:
            #Combine images q in a QpixMap and set it to self.image_label_result
            qq = QPixmap.fromImage(q).scaled(700, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label_result.setPixmap(qq)

    
    def run_mod_orb(self):
        self.algo_label.setText("Running Modified ORB")
            
        '''
        TEST 1
        AMADEO-BAUTISTA-LINGAD algorithm

        (ABL) Image Preprocessing + Bayesian Opt + ORB + FLANN + (MAGSAC++)
        '''
        self.query_image = cv2.imread(self.image_path)
        self.query_filename = self.image_path
        self.test_imgs = load_images_from_folder(r"C:\Users\johnl\Codes\ORBv2\test")

        matches, visuals = mod_orb.match(
            query_image=self.query_image, 
            query_filename=self.query_filename, 
            test_images=self.test_imgs
        ) 


        # for v in visuals:
        #     ind = visuals.index(v[5]) 
        #     visuals[ind] = "MOdified ORB"

        # print(visuals)
        self.visual_container = visuals
        
        for match in matches:
            total_matches = match[4]
            good_matches = match[2]
            good_match_percentage = (good_matches / total_matches) * 100

            # print("Verbose info:\t", match)
            print("\nAMADEO-BAUTISTA-LINGAD ORB algorithm")
            print("(OBJ2) Total matches:\t\t", total_matches)                    # OBJ 2
            print("(OBJ1) Good matches:\t\t", good_matches)                      # OBJ 1
            print("(OBJ3) Good match pecentage:\t", good_match_percentage)       # OBJ 3
            print("+++++++++++++++")
            print()

        # Convert BGR to RGB format for displaying in PyQt
        image = cv2.cvtColor(visuals[0][1], cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = channel * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # self.image_label_matched.setPixmap(QPixmap.fromImage(q_image).scaled(700, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # concatenate an image below the self.image_label_matched with another image
        mod_orb.save(visuals)
        q_imgs= []
        for v in visuals:
            # Read and convert the image using OpenCV
            image = cv2.imread(f"C:/Users/johnl/Codes/ORBv2/results/matched_img_{v[6]}")
            if image is not None:
                # Convert BGR to RGB format for displaying in PyQt
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channel = image.shape
                bytes_per_line = channel * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                q_imgs.append(q_image)

        qq = None
        for q in q_imgs:
            #Combine images q in a QpixMap and set it to self.image_label_result
            qq = QPixmap.fromImage(q).scaled(700, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label_result.setPixmap(qq)

    def save_results(self):
        
        mod_orb.save(self.visual_container)

if __name__ == "__main__": 
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec())
