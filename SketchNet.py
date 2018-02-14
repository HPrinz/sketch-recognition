import datetime
import pickle
import pprint
import time

from keras.utils import np_utils

import plot_utils
from SketchData import SketchData
from cnnmodel import CNNModel

pp = pprint.PrettyPrinter(indent=4)
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d_%H:%M:%S')

sketchdata = SketchData()

quickdraw = True

if quickdraw:
    X_train, y_train = sketchdata.get_training_data(True, "./quickdraw-train/*.npy", False)
    X_test, y_test = sketchdata.get_training_data(True, "./quickdraw-train/*.npy", False)
    # image size
    img_rows, img_cols = 28, 28
    nb_classes = 15
else:
    X_train, y_train = sketchdata.get_training_data(False, "./tu-train-small/**", False)
    X_test, y_test = sketchdata.get_training_data(False, "./tu-test-small/**", False)
    # image size
    img_rows, img_cols = 150, 150
    nb_classes = 16

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# uncomment for debugging
# show 9 grayscale images as examples of the data set
# ------- start show images ----------
# import sys
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title("Class {}".format(y_train[i]))
#
# plt.show()
# sys.exit()
# ------- end show images ----------

# converts a class vector (list of labels in one vector (as for SVM)
# to binary class matrix (one-n-encoding)
# pp.pprint(y_train)
Y_train = np_utils.to_categorical(y_train, nb_classes)
# pp.pprint(Y_train)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# we need to reshape the input data to fit keras.io input matrix format
X_train, X_test = CNNModel.reshape_input_data(X_train, X_test, img_rows, img_cols)

# hyperparameter
nb_epoch = 5
batch_size = 128

model = CNNModel.load_model(nb_classes, img_rows, img_cols)

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

# pickle.dump(model, open("results-cnn/" + timestamp + '.sav', 'wb'))

print("Test score: " + str(score[0]))
print("Test accuracy: " + str(score[1]))

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
plot_utils.plot_result_examples(model, X_test, y_test, img_rows, img_cols, timestamp)
