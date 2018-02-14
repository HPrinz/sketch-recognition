from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation


class FashionModel:

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
        input_shape = FashionModel.load_inputshape(img_rows, img_cols)

        model = Sequential()
        model.add(Convolution2D(input_shape=input_shape, data_format='channels_last', filters=32, kernel_size=(3, 3),
                                padding="same", activation="relu"))
        model.add(Convolution2D(kernel_size=(3, 3), filters=32, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        #model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=classes, activation="softmax"))

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
