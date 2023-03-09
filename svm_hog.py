# Importing the necessary modules:
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog

import joblib
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import yaml

class HOG_SVM():
    def __init__(self, svm_path):
        
        # define parameters of HOG feature extraction
        self.__model = LinearSVC(max_iter = 5000)
        self.__orientations = 9
        self.__pixels_per_cell = (8, 8)
        self.__cells_per_block = (2, 2)
        self.__num_neg_class = 1

    def train_process(self, data_path, data_pos_dir, data_neg_dir):
        pos_path = data_path + data_pos_dir
        neg_path = data_path + data_neg_dir

        filenames_pos = [join(pos_path, f) for f in listdir(pos_path) if isfile(join(pos_path, f))]
        filenames_neg = [join(neg_path, f) for f in listdir(neg_path) if isfile(join(neg_path, f))]
        
        labels = []
        data = []

        for i, fname in enumerate(filenames_pos):
            # The name of data file (npy) is its label class
            base    = os.path.basename(fname)
            cl      = os.path.splitext(base)[0]
            print("Loading ", fname, "... as class ", cl)

            # Load the data (npy)
            # The data is the list of images which are stored in .npy
            fn = np.load(fname)

            for f in fn:
                # Do the feature extraction with image by HOG
                fd = hog(f, self.__orientations, self.__pixels_per_cell, self.__cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
                
                # Append to the data and label lists for later training
                data.append(fd)
                labels.append(cl)

        for fname in filenames_neg:
            # The name of data file (npy) is its label class
            base    = os.path.basename(fname)
            cl      = os.path.splitext(base)[0]
            print("Loading ", fname, "... as class ", cl)

            # Load the data (npy)
            # The data is the list of images which are stored in .npy
            fn = np.load(fname)

            for f in fn:
                # Do the feature extraction with image by HOG                        
                fd = hog(f, self.__orientations, self.__pixels_per_cell, self.__cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor

                # Append to the data and label lists for later training
                data.append(fd)
                labels.append(cl)

        # encode the labels, converting them from strings to integers (if any)
        le = LabelEncoder()
        labels = le.fit_transform(labels)

        # Partitioning the data into training and testing splits, using 80%
        # of the data for training and the remaining 20% for testing
        (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.01, random_state=42)

        # Train the linear SVM
        print("... ... Training Linear SVM classifier ... !!!")
        self.__model.fit(trainData, trainLabels)
        print("... ... DONE training Linear SVM classifier !!!")

    def load_model(self, model_name = "svm.npy", model_info_name = "svm.yaml"):
        # Files exist?
        if os.path.isfile(model_name):
            if os.path.isfile(model_info_name):
                # Calib files
                with open (model_info_name, "r") as f:
                    info_file = yaml.safe_load(f)

                # Get the number of negative class
                # The application is just binary checking: Successful or Failed
                # Use number of negative class as the threshold to make decision
                self.__num_neg_class = info_file["info"]["num_neg_class"]

                # Load the model from weight
                self.__model = joblib.load(model_name)

                return True
            else:
                return False
        else:
            return False

    def save_model(self, modelName = "svm.npy"):
        # Save the model:
        joblib.dump(self.__model, modelName)

    def predict_process(self, img):
        # Do the feature extraction with image by HOG 
        fd = hog(img, self.__orientations, self.__pixels_per_cell, self.__cells_per_block, block_norm='L2', feature_vector=True) 
        fd = fd.reshape(1, -1)

        # Predict by model
        prediction = self.__model.predict(fd)

        # print(prediction[0])
        
        # Decision making
        if prediction[0] >= self.__num_neg_class:
            # print("[SVM classification] Class: Successful")
            return 1
        else:
            # print("[SVM classification] Class: Failed")
            return 0
