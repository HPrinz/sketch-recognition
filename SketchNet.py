import datetime
import pickle
import pprint
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from keras.utils import np_utils

import plot_utils
from SketchData import SketchData
from cnnmodels.SketchANetModel import SketchANetModel
from cnnmodels.SketchANetModelAdapted import SketchANetModelAdapted
from cnnmodels.FashionModel import FashionModel


class SketchNet:

    @staticmethod
    def write_results(self, timestamp, history, nb_epoch, batch_size, score, model, x_test, y_test, img_rows, img_cols):
        filename = 'results-cnn/' + str(timestamp) + '_report_CNN.txt'
        with open(filename, "w") as fh:
            fh.write("Timestamp: " + str(timestamp) + "\n")
            fh.write("Epochs: " + str(nb_epoch) + "\n")
            fh.write("Batch Size: " + str(batch_size) + "\n")
            fh.write("Test score: " + str(score[0]) + "\n")
            fh.write("Test accuracy: " + str(score[1]) + "\n")
            fh.write("Test Model:\n")
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write("\n")
            fh.close()

        plot_utils.plot_model_history(history, timestamp)
        plot_utils.plot_result_examples(model, x_test, y_test, img_rows, img_cols, timestamp)

    def train(self):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d_%H:%M:%S')

        scaler = StandardScaler()
        x_train = scaler.fit_transform(self.x_train)
        x_test = scaler.transform(self.x_test)

        # image size
        img_rows, img_cols = 28, 28

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        print("Training matrix shape", x_train.shape)
        print("Testing matrix shape", x_test.shape)

        # converts a class vector (list of labels in one vector (as for SVM)
        # to binary class matrix (one-n-encoding)
        y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        y_test = np_utils.to_categorical(self.y_test, self.nb_classes)

        # we need to reshape the input data to fit keras.io input matrix format
        x_train, x_test = self.modeltype.reshape_input_data(x_train, x_test, img_rows, img_cols)

        # hyperparameter
        nb_epoch = 5
        batch_size = 128

        model = self.modeltype.load_model(self.nb_classes, img_rows, img_cols)

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,
                            validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)

        # pickle.dump(model, open("results-cnn/" + timestamp + '.sav', 'wb'))

        print("Test score: " + str(score[0]))
        print("Test accuracy: " + str(score[1]))

        self.write_results(timestamp, history, nb_epoch, batch_size, score, model, x_test, y_test, img_rows, img_cols)

    def __init__(self, modeltype, quickdraw=False):
        # np.set_printoptions(threshold=np.nan)
        # pp = pprint.PrettyPrinter(indent=4)

        sketchdata = SketchData()

        self.modeltype = modeltype

        if quickdraw:
            self.x_train, self.y_train = sketchdata.get_training_data(True, "./quickdraw-train/*.npy", False)
            self.x_test, self.y_test = sketchdata.get_training_data(True, "./quickdraw-test/*.npy", False)
            self.nb_classes = 15
        else:
            self.x_train, self.y_train = sketchdata.get_training_data(False, "./tu-train-small/**/", False)
            self.x_test, self.y_test = sketchdata.get_training_data(False, "./tu-test-small/**/", False)
            self.nb_classes = 40
