from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation


class SketchANetModel:
    @staticmethod
    def load_inputshape(img_rows, img_cols):
        return img_rows, img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test, img_rows, img_cols):
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        return x_train, x_test

    @staticmethod
    def load_model(classes=10, img_rows=28, img_cols=28):
        input_shape = SketchANetModel.load_inputshape(img_rows, img_cols)

        model = Sequential()
        # 1 Input: 225x225 Output: 71x71
        model.add(Convolution2D(input_shape=input_shape, data_format='channels_last', strides=(1, 1), filters=64,
                                kernel_size=(2, 2), padding="same", activation="relu"))
        # Inout 71x71 Output: 35x35
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # 2 Input: 35x35 Output: 31x31
        model.add(Convolution2D(kernel_size=(3, 3), filters=128, strides=(1, 1), activation="relu", padding="same"))
        # Input: 31x31 Output: 15x15s
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # Input: 15x15 Output: 15x15
        model.add(Convolution2D(kernel_size=(3, 3), filters=256, strides=(1, 1), activation="relu", padding="valid"))
        model.add(Convolution2D(kernel_size=(3, 3), filters=256, strides=(1, 1), activation="relu", padding="valid"))
        model.add(Convolution2D(kernel_size=(3, 3), filters=256, strides=(1, 1), activation="relu", padding="valid"))
        # Input: 15x15 Output: 7x7
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # Input: 7x7 Output: 1x1
        model.add(Convolution2D(kernel_size=(7, 7), filters=512, strides=(1, 1), activation="relu", padding="same"))
        # Input: 1x1 Output: 1x1
        model.add(Dropout(rate=0.5))

        model.add(Convolution2D(kernel_size=(1, 1), filters=512, strides=(1, 1), activation="relu", padding="same"))
        model.add(Dropout(rate=0.5))

        model.add(Convolution2D(kernel_size=(1, 1), filters=512, strides=(1, 1), activation="relu", padding="same"))

        model.add(Flatten())
        model.add(Dense(units=classes, activation="softmax"))

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
