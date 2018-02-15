import cv2
import glob
import os
import pprint
import datetime
import time
from MidpointNormalize import *

pp = pprint.PrettyPrinter(indent=4)


class SketchData:
    """Support Vector Machine for Sketch recognition"""

    def __init__(self, size, step):

        self.keypoints = self.create_keypoints(size, size, step)

        # histogram orientations: 4 neighbors horizontal * 4 neighbors vertical * 8 directions
        num_entry_per_keypoint = 4 * 4 * 8

        self.descriptor_length = num_entry_per_keypoint * len(self.keypoints)

        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d_%H:%M:%S')

        self.categories = []

        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()

    def get_keypoints(self):
        return self.keypoints

    def load_images(self, train_path):
        # read categories by folders
        self.categories = sorted(glob.glob(train_path))
        categories_and_images = []
        num_train_images = 0

        """read all image paths in each category folder"""
        for cat in self.categories:
            category_name = os.path.basename(os.path.normpath(cat))
            images_in_cat = glob.glob(cat + '/*.png')
            num_train_images += len(images_in_cat)
            categories_and_images.append((category_name, images_in_cat))

        return categories_and_images, num_train_images

    def load_images_google(self, train_path):
        categories = sorted(glob.glob(train_path))
        categories_and_images = []
        num_train_images = 0

        pprint.pprint(categories)

        for test_filename in categories:
            images = np.load(test_filename)

            images_formatted = []
            for image in images:
                image_pxl = image.reshape(28, 28)
                image_pxl = np.invert(image_pxl)
                images_formatted.append(image_pxl)

            num_train_images += len(images_formatted)
            categories_and_images.append((test_filename, images_formatted))

        return categories_and_images, num_train_images

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

    def get_training_data(self, google, path, sift=True):
        if google:
            categories_and_images, num_train_images = self.load_images_google(path)
        else:
            categories_and_images, num_train_images = self.load_images(path)

        print("loaded %d images" % num_train_images)

        if sift:
            deslen = self.descriptor_length
        else:
            deslen = 28 * 28

        # create y_train vector containing the labels as integers
        y_train = np.zeros(num_train_images, dtype=int)

        # x_train matrix containing decriptors as vectors
        x_train = np.zeros((num_train_images, deslen))

        index_img = 0
        index_cat = 0

        for (cat, image_filenames) in categories_and_images:
            for image in image_filenames:
                if google:
                    image_read = image
                else:
                    image_read = cv2.imread(image, 0)

                if sift:
                    des = self.create_sift_descriptors_for_image(image_read)
                else:
                    des = image_read

                # each descriptor (set of features) need to be flattened in one vector
                x_train[index_img] = des.flatten()
                y_train[index_img] = index_cat
                index_img = index_img + 1
            index_cat = index_cat + 1

        return x_train, y_train

    def get_name_for_category(self, category):
        return self.categories[category]
