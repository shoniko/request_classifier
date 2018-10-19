"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras as K

FLAGS = tf.flags.FLAGS

class CNN():
    def __init__(self, params):
        return self.buildModel(params)

    def buildModel(self, cfg):
        urlInput = K.Input(shape=(None, cfg["input_size"]), name="urls_input", dtype="int64")
        print("Input shape")
        print(urlInput.shape)
        x = K.layers.Embedding(cfg["alphabet_size"],
            cfg["embedding_size"],
            input_length=cfg["input_size"])(urlInput)

        print("After embedding shape")
        print(x.shape)
        # Convolution layers
        for cl in cfg["conv_layers"]:
            x = K.layers.Convolution1D(cl[0], cl[1])(x)
            print("After conv shape")
            print(x.shape)

            x = K.layers.ThresholdedReLU(cfg["threshold"])(x)
            if cl[2] != -1:
                x = K.layers.MaxPooling1D(cl[2])(x)
        x = K.layers.Flatten()(x)
        
        # Fully connected layers
        for fc in cfg["fc_layers"]:
            x = K.layers.Dense(fc)(x)
            x = K.layers.ThresholdedReLU(cfg["threshold"])(x)
            x = K.layers.Dropout(cfg["dropout"])(x)
        # Output layer
        predictions = K.layers.Dense(1, activation='sigmoid')(x)
        # Build and compile model
        self.model = K.Model(inputs=urlInput, outputs=predictions)
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
