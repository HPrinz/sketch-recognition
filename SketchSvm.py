import numpy as np
import cv2
import glob
import os
import pprint
import datetime
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
from MidpointNormalize import *
from PIL import Image
# import matplotlib.pyplot as plt
from sklearn_evaluation import plot
from sklearn.preprocessing import StandardScaler
from SketchData import SketchData

pp = pprint.PrettyPrinter(indent=4)


class SketchSvm:
    """Support Vector Machine for Sketch recognition"""

    def __init__(self, size, step):

        self.sketchdata = SketchData(size, step)

        self.keypoints = self.sketchdata.get_keypoints()
        print("created " + str(len(self.keypoints)) + " keypoints")

        # histogram orientations: 4 neighbors horizontal * 4 neighbors vertical * 8 directions
        num_entry_per_keypoint = 4 * 4 * 8

        self.descriptor_length = num_entry_per_keypoint * len(self.keypoints)

        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d_%H:%M:%S')

        self.scaler = StandardScaler()

    def fit_scaler(self, quickdraw, trainpath):
        x_train, y_train = self.sketchdata.get_training_data(quickdraw, trainpath, sift=True)
        self.scaler.fit(x_train)

    def load_model(self, model_file):
        if not model_file:
            print("model name has to be provided")
            exit(0)

        if not os.path.isfile(model_file):
            print("model not found")
            exit(0)

        return pickle.load(open(model_file, 'rb'))

    def draw_heatmap(self, google, model, params, kernel):
        if google:
            path = "results-quickdraw/"
        else:
            path = "results/"
        fig = plot.grid_search(model.grid_scores_, change=params)
        fig.get_figure().savefig(path + self.timestamp + "_" + kernel + ".pdf")

        # fig = plot.grid_search(model.grid_scores_, change=params, subset={"kernel": "rbf"})
        # fig.get_figure().savefig("results/" + self.timestamp + "rbf.pdf")

    def save_model(self, google, model):
        if google:
            save_location = 'models-quickdraw/'
        else:
            save_location = 'models/'
        pickle.dump(model, open(save_location + self.timestamp + '.sav', 'wb'))

    def train(self, quickdraw, path, c_range, gamma_range, kernel, save=True):
        x_train, y_train = self.sketchdata.get_training_data(quickdraw, path, sift=False)
        x_train = self.scaler.fit_transform(x_train)

        # parameters = {'C': c_range, "gamma": gamma_range, 'kernel': ['linear', 'rbf']}
        parameters = {'C': c_range, "gamma": gamma_range}

        # train a SVM classifier, specifically a GridSearchCV in our case
        clf = GridSearchCV(svm.SVC(kernel=kernel), parameters)
        clf.fit(x_train, y_train)

        # save model
        if save:
            self.save_model(quickdraw, clf)

        self.draw_heatmap(quickdraw, clf, ('C', 'gamma'), kernel)

        return clf

    def test_image(self, image, model):
        des = self.sketchdata.create_sift_descriptors_for_image(image)

        test_descriptor = np.zeros((1, self.descriptor_length))
        test_descriptor[0] = des.flatten()

        test_descriptor = self.scaler.transform(test_descriptor)

        label = model.best_estimator_.predict(test_descriptor)

        return label[0]

    def test_model(self, model, testpath):
        # 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
        # the same way we did for the training (except for a single image now) and use .predict()
        # to classify the image

        test_images = glob.glob(testpath)

        for test_filename in test_images:
            image_read = cv2.imread(test_filename, 0)
            label = self.test_image(image_read, model)
            # output the identified class
            print(
                "label of " + test_filename + " is predicted as \"" + self.sketchdata.get_name_for_category(
                    label) + "\"")

    def test_google(self, model, googletestpath):
        test_images = glob.glob(googletestpath)

        for test_filename in test_images:
            images = np.load(test_filename)

            for image in images:
                image_pxl = image.reshape(28, 28)
                image_pxl = np.invert(image_pxl)
                im = Image.fromarray(image_pxl)
                im.show()
                label = self.test_image(image_pxl, model)
                # output the identified class
                print("label of " + test_filename + " is predicted as \"" + self.sketchdata.get_name_for_category(
                    label) + "\"")
