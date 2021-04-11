# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file models.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details This module contains a set of functions to generate Machine Learning models such as Feed-forward Neural
Networks, Recurrent Neural Networks (RNN), Convolutional Neural Networks (CNN) and more.
"""
import Yomchi.Environment as Env

import tensorflow as tf


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def MLP(layers, units_per_layer, dropout=None):

    Env.print_text(f"Creating a dense feed-forward Neural Network model with {layers} layers, using {units_per_layer} "
                   f"units per layer. \nThe activation function in the hidden layers is 'ReLu' and the output layer "
                   f"has only 1 unit (the predicted angle).")
    model = tf.keras.Sequential()

    # First layer (need to specify the input size)
    model.add(tf.keras.layers.Dense(
        units=units_per_layer,
        # input_shape=(window_size,),
        # kernel_initializer='he_normal',
        # bias_initializer='zeros',
        activation=tf.nn.relu))

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    # Other hidden layers
    for n in range(1, layers):
        model.add(tf.keras.layers.Dense(
            units=units_per_layer,
            # kernel_initializer='he_normal',
            # bias_initializer='zeros',
            activation=tf.nn.relu))

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    # Output layer
    model.add(tf.keras.layers.Dense(
        units=1
        # kernel_initializer='glorot_normal',
        # bias_initializer='zeros',
        # activation=tf.nn.softmax
        ))
    return model


def compile_and_fit(model, train, val, epochs=20, optimizer="Adam", patience=2):

    Env.print_text(f"Compiling the input model {model.name} using '{optimizer}' optimizer with {epochs} epochs. The"
                   f"loss function is the MSE and metric to evaluate improvement is the MAE.")

    if optimizer == "Adagrad":
        optimizer = tf.optimizers.Adagrad()
    elif optimizer == "Adadelta":
        optimizer = tf.keras.optimizers.Adadelta()
    elif optimizer =="Adam":
        optimizer = tf.optimizers.Adam()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(train, epochs=epochs,
                        validation_data=val,
                        callbacks=[early_stopping])
    return history