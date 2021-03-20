import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

""" The autoencoder class is consists of two main parts: encoder and decoder.

    To create an autoencoder, need to spacify the input dimansion.

    To train the autoencoder, need to give train set and test set.
    Training aim is to minimize the mse given to the autoencoder as input and the output of
    the autoencoder.

    To get a decoded image from the autoencoder, give the image to perform the operation on.
    Returns the reconstructions also in mse - as builded on.
"""
class Autoencoder(Model):
    def __init__(self, input_dim):
        #Encoding
        inp = Input(shape=(input_dim,))
        encoder = Dense(input_dim/2, activation="relu", activity_regularizer=regularizers.l1(10e-7))(inp)
        encoder = Dense(input_dim/4, activation="relu", kernel_regularizer=regularizers.l2(10e-7))(encoder)
        #decoding
        decoder = Dense(input_dim/2, activation="relu", kernel_regularizer=regularizers.l2(10e-7))(encoder)
        decoder = Dense(input_dim, activation="sigmoid", kernel_regularizer=regularizers.l2(10e-7))(decoder)

        self._model = Model(inputs=inp, outputs=decoder)
        self._model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    
    def train(self, input_train, input_test):
        history = self._model.fit(input_train, input_train, epochs=200, batch_size=256, shuffle=True,
                                validation_data=(input_test, input_test))

        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.legend()

    
    def get_decoded_image(self, test_image):
        reconstructions = self._model.predict(test_image[None])[0] # I'm not sure why, but the None value makes it work on one dimensional array..
        return reconstructions