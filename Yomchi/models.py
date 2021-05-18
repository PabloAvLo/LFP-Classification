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

    # Shape: (time, features) => (time*features)
    model.add(tf.keras.layers.Flatten())

    # First layer (need to specify the input size)
    model.add(tf.keras.layers.Dense(
        units=units_per_layer,
        #input_shape=(32, 125, 1),
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

    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    model.add(tf.keras.layers.Reshape([1, -1]))

    #Env.print_text(f'Model Output shape: {model.output_shape}')

    return model

def CNN(inputs, units_per_layer):
    Env.print_text(f"Creating a Convolutional Neural Network model with one 1D convolutional layer, using "
                   f"{units_per_layer} filters and units in the following dense layer. \nThe activation function in "
                   f"the dense layer is 'ReLu' and the output dense layer has only 1 unit (the predicted angle).")

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=units_per_layer,
                               kernel_size=(inputs,),
                               activation='relu'),
        tf.keras.layers.Dense(units=units_per_layer, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    return conv_model

def LSTM(units_per_layer):
    Env.print_text(f"Creating a Long Short-term Memory (LSTM) Neural Network using {units_per_layer} units per layer."
                   f"\nThis network only outputs the final timestamp, giving the model time to warm up its internal  "
                   f"state before making a single prediction.")

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(units_per_layer, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return lstm_model

def compile_and_fit(model, train, val, epochs=20, optimizer="Adam", patience=2):

    Env.print_text(f"Compiling the input model {model.name} using '{optimizer}' optimizer with {epochs} epochs. The "
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