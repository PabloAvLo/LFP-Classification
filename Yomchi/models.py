# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file models.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date July, 2021
@details This module contains a set of functions to generate Machine Learning models such as Feed-forward Neural
Networks, Long Short-Term Memory Neural Network (LSTM), Convolutional Neural Networks (CNN) and more.
"""
import Yomchi.Environment as Env
import tensorflow as tf


def mlp(layers, units_per_layer, dropout=None):
    """
    Defines a classical neural network sometimes called: Multi-Layer Perceptron. It is Feedforward and fully-connected,
    with the specified parameters. The activation function of the hidden layers is ReLU, and the activation of the
    output is linear.
    @param layers: Number of hidden layers (besides the input and outputs ones).
    @param units_per_layer: Number of neurons per layer. Applies to all layers except for the last one which has only 1:
    the predicted angle.
    @param dropout: A value between 0 and 1 of neuron's results to discard of the training. This regularization method
    will be added to the model. One after the input layer and one before the output layer.
    @return model: The model of the MLP created for later usage as the predictor.
    """

    Env.print_text(f"Creating a fully-connected feed-forward Neural Network model with {layers} layers, using "
                   f"{units_per_layer} units per layer. \nThe activation function in the hidden layers is 'ReLu' and "
                   f"the output layer has only 1 unit (the predicted angle).")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())  # Shape: (time, features) => (time*features)
    model.add(tf.keras.layers.Dense(units=units_per_layer, activation=tf.nn.relu))  # Input layer
    model.add(tf.keras.layers.Dropout(dropout))

    for n in range(1, layers):  # Other hidden layers
        model.add(tf.keras.layers.Dense(units=units_per_layer, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=1))  # Output layer
    model.add(tf.keras.layers.Reshape([1, -1]))  # Add back the time dimension. Shape: (outputs) => (1, outputs)

    return model


def cnn(inputs, units_per_layer, dropout):
    """
    Creates a model of a Convolutional Neural Network with a Conv1D as the input layer, one dense as the only hidden
    layer and another dense with only 1 neuron as the output layer. The first two layer has ReLU activation and the last
    one uses a linear activation function.S
    @param inputs: Number of inputs of the network. It is used as the kernel size with the intention that the 1D
    convolutional layer outputs a single value throughout the specified number of filters.
    @param units_per_layer: Specified the number of neurons of the hidden dense layer, which has to match with the
    number of filters that will output a result in the 1D convolutional input layer.
    @param dropout: A value between 0 and 1 of neuron's results to discard of the training. This regularization method
    will be added to the model as a layer the output layer.
    @return conv_model: The model of the CNN created for later usage as the predictor.
    """
    Env.print_text(f"Creating a Convolutional Neural Network model with one 1D convolutional layer, using "
                   f"{units_per_layer} filters and units in the following dense layer. \nThe activation function in "
                   f"the dense layer is 'ReLU' and the output dense layer has only 1 unit (the predicted angle).")

    conv_model = tf.keras.Sequential()
    conv_model.add(tf.keras.layers.Conv1D(filters=units_per_layer, kernel_size=(inputs,), activation='relu'))
    conv_model.add(tf.keras.layers.Dense(units=units_per_layer, activation='relu'))
    conv_model.add(tf.keras.layers.Dropout(dropout))
    conv_model.add(tf.keras.layers.Dense(units=1))

    return conv_model


def lstm(units_per_layer, dropout):
    """
    Creates a model of a Long Short-Term Memory Neural Network as the input layer, one dense with only 1 neuron as the
    output layer. The activation functions of the LSTM layer are the regular ones for each of it's gates.
    @param units_per_layer: Number of inputs of the network.
    @param dropout: A value between 0 and 1 of neuron's results to discard of the training. If provided,this
    regularization method will be to the LSTM Layer.
    @return lstm_model: The model of the LSTM created for later usage as the predictor.
    """

    Env.print_text(f"Creating a Long Short-term Memory (LSTM) Neural Network using {units_per_layer} units per layer."
                   f"\nThis network only outputs the final timestamp, giving the model time to warm up its internal  "
                   f"state before making a single prediction.")

    # Shape [batch, time, features] => [batch, time, lstm_units]
    lstm_model = tf.keras.models.Sequential()
    lstm_model.add(tf.keras.layers.LSTM(units_per_layer, return_sequences=False, dropout=dropout))
    lstm_model.add(tf.keras.layers.Dense(units=1))  # Shape => [batch, time, features]

    return lstm_model


def compile_and_fit(model, train, val, epochs=20, patience=2):
    """
    Fits the provided training data in the specified model and train's it. The validation data is used to compare the
    performance of the model against unknown data. MSE is used as the cost function to train the model and MAE as the
    performance evaluation metric.
    @param model: Model to train. It can be a MLP, CNN or LSTM, among others.
    @param train: The dataset for training the model. Must have the shape: (batch, time, features)
    @param val: The dataset for validating the model. Must have the shape: (batch, time, features)
    @param epochs: Number of iteration over the entire set of data (all the batches).
    @param patience: Number of epochs to wait for improvement in the metrics. If there is no notorious improvement in
    the performance of the validation set after the 'patience' epochs, the training will stop at this point.
    @return history: The results of the training.
    """

    Env.print_text(f"Compiling the input model {model.name} with {epochs} epochs. The "
                   f"loss function is the MSE and metric to evaluate improvement is the MAE.")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),  # tf.losses.mean_absolute_percentage_error()
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(train, epochs=epochs,
                        validation_data=val,
                        callbacks=[early_stopping])

    return history
