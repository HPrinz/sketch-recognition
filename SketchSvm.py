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

pp = pprint.PrettyPrinter(indent=4)

class SketchSvm:
    """Support Vector Machine for Sketch recognition"""

    def __init__(self):

        self.keypoints = self.create_keypoints(150, 150, 30)
        print("created " + str(len(self.keypoints)) + " keypoints")

        # histogram orientations: 4 neighbors horizontal * 4 neighbors vertical * 8 directions
        num_entry_per_keypoint = 4 * 4 * 8

        self.descriptor_length = num_entry_per_keypoint * len(self.keypoints)

        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d_%H:%M:%S')

        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.categories = []

        self.categories_and_images = []
        self.num_train_images = 0

        self.scaler = StandardScaler()

    def load_images(self, train_path):
        # read categories by folders
        self.categories = glob.glob(train_path)
        # categories = glob.glob('./img/elephant')

        """read all image paths in each category folder"""
        for cat in self.categories:
            category_name = os.path.basename(os.path.normpath(cat))
            images_in_cat = glob.glob(cat + '/*.png')
            self.num_train_images = self.num_train_images + len(images_in_cat)
            self.categories_and_images.append((category_name, images_in_cat))

    def load_images_google(self, train_path):
        self.categories = glob.glob(train_path)
        self.num_train_images = 0

        pprint.pprint(self.categories)

        for test_filename in self.categories:
            images = np.load(test_filename)

            images_formatted = []
            for image in images:
                image_pxl = image.reshape(28, 28)
                image_pxl = np.invert(image_pxl)
                images_formatted.append(image_pxl)

            self.num_train_images = self.num_train_images + len(images_formatted)
            self.categories_and_images.append((test_filename, images_formatted))

    def load_model(self, model_file):
        if not model_file:
            print("model name has to be provided")
            exit(0)

        if not os.path.isfile(model_file):
            print("model not found")
            exit(0)

        return pickle.load(open(model_file, 'rb'))

    def create_keypoints(self, w, h, keypoint_size):
        """
        creating keypoints on grid for later image segmentation
        :param w: width of grid
        :param h: height of grid
        :param keypoint_size: keypoint size
        :return: array of kreypoints
        """
        keypoints_list = []

        for x in range(0, w, keypoint_size):
            for y in range(0, h, keypoint_size):
                keypoints_list.append(cv2.KeyPoint(x, y, keypoint_size))

        return keypoints_list

    def create_sift_descriptors_for_image(self, image):
        # compute one descriptor for each image
        keyp, descr = self.sift.compute(image, self.keypoints)
        return descr

    def draw_heatmap(self, google, model, params):
        if google:
            path = "results/"
        else:
            path = "results-quickdraw/"
        fig = plot.grid_search(model.grid_scores_, change=params)
        fig.get_figure().savefig(path + self.timestamp + "rbf.pdf")

        # fig = plot.grid_search(model.grid_scores_, change=params, subset={"kernel": "rbf"})
        # fig.get_figure().savefig("results/" + self.timestamp + "rbf.pdf")

    def train_model(self, path, c_range, gamma_range, kernel="linear"):
        self.load_images(path)
        print("loaded %d images" % self.num_train_images)
        return self.train(False, c_range, gamma_range, kernel)

    def train_model_google(self, path, c_range, gamma_range, kernel="linear"):
        self.load_images_google(path)
        print("loaded %d images" % self.num_train_images)
        return self.train(True, c_range, gamma_range, kernel)

    def save_model(self, google, model):
        if google:
            save_location = 'models-quickdraw/'
        else:
            save_location = 'models/'
        pickle.dump(model, open(save_location + self.timestamp + '.sav', 'wb'))

    def get_training_data(self, google):
        # create y_train vector containing the labels as integers
        y_train = np.zeros(self.num_train_images, dtype=int)

        # x_train matrix containing decriptors as vectors
        x_train = np.zeros((self.num_train_images, self.descriptor_length))

        index_img = 0
        index_cat = 0

        for (cat, image_filenames) in self.categories_and_images:
            for image in image_filenames:
                if google:
                    image_read = image
                else:
                    image_read = cv2.imread(image, 0)

                des = self.create_sift_descriptors_for_image(image_read)

                # each descriptor (set of features) need to be flattened in one vector
                x_train[index_img] = des.flatten()
                y_train[index_img] = index_cat

                index_img = index_img + 1
            index_cat = index_cat + 1
        x_train = self.scaler.fit_transform(x_train)
        return x_train, y_train

    def train(self, google, c_range, gamma_range, kernel, save=True):
        x_train, y_train = self.get_training_data(google)

        # parameters = {'C': c_range, "gamma": gamma_range, 'kernel': ['linear', 'rbf']}
        parameters = {'C': c_range, "gamma": gamma_range}

        # train a SVM classifier, specifically a GridSearchCV in our case
        clf = GridSearchCV(svm.SVC(kernel), parameters)
        clf.fit(x_train, y_train)

        # save model
        if save:
            self.save_model(google, clf)

        self.draw_heatmap(google, clf, ('C', 'gamma'))

        return clf

    def test_image(self, image, model):
        des = self.create_sift_descriptors_for_image(image)

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
            print("label of " + test_filename + " is predicted as \"" + self.categories[label] + "\"")

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
                print("label of " + test_filename + " is predicted as \"" + self.categories[label] + "\"")
