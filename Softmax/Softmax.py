# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file Softmax.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details Feed-forward Neural Network using Softmax algorithm in the output layer.
"""

import Yomchi.Environment as Env
import Yomchi.preprocessing as data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

## <h1> Feed-Forward Neural Network </h1>
# <h2> Experiment Setup </h2>
# <ul>
# <li> Initialize Environment
# <li> Initialize Tensorflow session.
# <li> Set seed for Numpy and Tensorflow
# <li> Specify run configuration parameters.
# <li> Specify session and methods parameters.
# <li> Specify data properties parameters.
# </ul>
# <ol>
Env.init_environment(True, enable_debug=True)

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

# Session and methods Parameters
#session = {"Number": "771", "LFP Data": data.LFP_771, "Angles Data": data.ANGLES_771}  # or
session = {"Number": "765", "LFP Data": data.LFP_765, "Angles Data": data.ANGLES_765}

interpolation = "SLERP"  # "linear" "quadratic" "cubic" "nearest" "SLERP"
sync_method = "Upsample Angles"  # "Upsample Angles" "Downsample LFPs"

# Data Properties
num_channels = data.EC014_41_NUMBER_OF_CHANNELS
rate_used = data.POSITION_DATA_SAMPLING_RATE
if sync_method == "Upsample Angles":
    rate_used = data.LFP_DATAMAX_SAMPLING_RATE

# Windowing properties
window_size = 32
batch_size = 32
shuffle_buffer = 1000
lfp_channel = 72
round_angles = False
base_angle = 0  # Unused if round_angles = False
offset_between_angles = 45  # Unused if round_angles = False

extra = {"Round angles to get discrete label": round_angles}
if round_angles:
    extra.update({"Angle labels starting from": str(base_angle) + "°",
                  "until 360° in steps of": str(offset_between_angles) + "°"})

parameters_dictionary = {"Recording Session Number": session["Number"],
                      "Interpolation Method": interpolation.title(),
                      "Synchronization Method": sync_method,
                      "Number of Recorded Channels": str(num_channels),
                      "Sampling Rate to Use": str(rate_used) + " Hz",
                      "Window Size": window_size,
                      "Batch size (# of windows)": batch_size,
                      "LFP Signal channel to use": lfp_channel,
                      "Shuffle buffer size": shuffle_buffer}

parameters_dictionary.update(extra)
Env.print_parameters(parameters_dictionary)

## <li> Step 1
# <ul>
# <li> Import LFP data.
# <li> Import angles data.
# </ul>
Env.step("Importing LFP signal data and position data.")

Env.print_text("Importing LFP data from session: " + session["Number"])
lfp_data = data.load_lfp_data(session["LFP Data"])

Env.print_text("Importing angles data from session: " + session["Number"])
angles_data = data.load_angles_data(session["Angles Data"])

## <li> Step 2
# <ul>
# <li> Synchronize the LFP signal sampling rate with the angles sampling rate by either downsample the
# <br> LFP data to match Angles sampling rate, or Upsample Angles data to reach a higher sampling rate
# </ul>
Env.step("Synchronize the LFP signal sampling rate with the angles sampling rate.")

if sync_method == "Downsample LFPs":
    Env.print_text("Downsample LFP data to match Angles sampling rate.")
    lfp_data = data.downsample_lfps(lfp_data, data.LFP_DATAMAX_SAMPLING_RATE, data.POSITION_DATA_SAMPLING_RATE)

elif sync_method == "Upsample Angles":
    Env.print_text("Upsample Angles data to reach a higher sampling rate.")
    angles_data = data.angles_expansion(angles_data, data.POSITION_DATA_SAMPLING_RATE, data.LFP_DATAMAX_SAMPLING_RATE)

## <li> Step 3
# <ul>
# <li> Interpolate angles data using a 'interpolation' approach.
# <li> Label data by concatenating LFPs and interpolated Angles in a single 2D-array.
# <li> Clean the labeled dataset from NaN values at the boundaries.
# </ul>
Env.step()

Env.print_text("Interpolate angles data using a " + interpolation + " approach.")
angles_data_interpolated = data.interpolate_angles(angles_data, interpolation)

Env.print_text("Label data by concatenating LFPs and interpolated Angles in a single 2D-array.")
labeled_data = data.add_labels(lfp_data, np.expand_dims(angles_data_interpolated, axis=1), round_angles,
                               base_angle, offset_between_angles)

Env.print_text("Clean the labeled dataset from NaN values at the boundaries.")
clean_dataset = data.clean_unsync_boundaries(labeled_data, False)

Env.print_text("Convert the data to a windowed series.")

## <li> Step 4
# <ul>
# <li> Split the data in training set and validation set.
# <li> Convert the training data to a windowed series.
# </ul>
Env.step()

(rows, columns) = clean_dataset.shape
total_length = int(rows * 0.01)
train_len = int(total_length * 0.8)
x_train = clean_dataset[:train_len, :]
x_valid = clean_dataset[train_len:total_length, :]

print("Training data shape: [", len(x_train), ":", len(x_train[0]), "]")
print("Validation data shape: [", len(x_valid), ":", len(x_valid[0]), "]")

Env.print_box("INITIALIZING TENSORFLOW")
windowed_data = data.channels_to_windows(x_train, lfp_channel, window_size, batch_size, shuffle_buffer)
windowed_validation_data = data.channels_to_windows(x_valid, lfp_channel, window_size, batch_size)

## <li> Step 5
# <ul>
# <li> Create model: Multi Layer Perceptron (MLP) with:
# </ul>
Env.step()

# Parameters
n_nodes_per_layer = 500
dropout_rate = 0.60
n_layers = 3
epochs = 200

# Create a neural network model
model = tf.keras.models.Sequential()

# First layer (need to specify the input size)
print("Adding layer with {} nodes".format(n_nodes_per_layer))
model.add(tf.keras.layers.Dense(
    units=n_nodes_per_layer,
    input_shape=(window_size,),
    activation=tf.nn.relu,
    kernel_initializer='he_normal',
    bias_initializer='zeros'))
model.add(tf.keras.layers.Dropout(dropout_rate))

# Other hidden layers
for n in range(1, n_layers):
    print("Adding layer with {} nodes".format(n_nodes_per_layer))
    model.add(tf.keras.layers.Dense(
        units=n_nodes_per_layer,
        activation=tf.nn.relu,
        kernel_initializer='he_normal',
        bias_initializer='zeros'))
    model.add(tf.keras.layers.Dropout(dropout_rate))

# Output layer
print("Adding layer with 1 node")
model.add(tf.keras.layers.Dense(
    units=360,
    activation=tf.nn.softmax,
    kernel_initializer='glorot_normal',
    bias_initializer='zeros'))

## <li> Step 6
# <ul>
# <li> Define the optimizer
# <li> Use the Mean Square Error (MSE) as Loss function.
# </ul>
Env.step()

# optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = tf.keras.optimizers.Adadelta(lr=0.001)
# optimizer = keras.optimizers.Adagrad(lr=0.01)

# Define cost function and optimization strategy
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['accuracy']
)

## <li> Step 7
# <ul>
# <li> Train the neural network
# </ul>
Env.step()

start_time = time.time()

history = model.fit(
    x=windowed_data,
    #x=x_train[:, :-1],
    #y=x_train[:, -1],
    epochs=epochs,
    #batch_size=batch_size,
    validation_data=windowed_validation_data,
    #validation_data=(x_valid[:, :-1], x_valid[:, -1]),
    verbose=2
    )

## <li> Step 8
# <ul>
# <li> Metrics
# </ul>
Env.step()

end_time = time.time()
print("Training time: {:.0f}:{:.0f} [m:s]".format((end_time - start_time)/60, (end_time - start_time) % 60))

# Find the best costs & metrics
test_accuracy_hist = history.history['val_accuracy']
best_idx = test_accuracy_hist.index(max(test_accuracy_hist))
print("Max test accuracy:  {:.4f} at epoch: {}".format(test_accuracy_hist[best_idx], best_idx))

trn_accuracy_hist = history.history['accuracy']
best_idx = trn_accuracy_hist.index(max(trn_accuracy_hist))
print("Max train accuracy: {:.4f} at epoch: {}".format(trn_accuracy_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

## <li> Step 9
# <ul>
# <li> Predict
# </ul>
Env.step()

print("----------------------------------------------")
prediction = model.predict(windowed_validation_data)

unique, counts = np.unique(prediction, return_counts=True)
print(dict(zip(unique, counts)))

"""
forecast = []

for time in range(total_length - window_size):
  forecast.append(model.predict(windowed_data[time:time + window_size][np.newaxis]))

forecast = forecast[train_len-window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(10, 6))

#ts.plot_series(time_valid, x_valid)
#ts.plot_series(time_valid, results)

mse = tf.keras.metrics.mean_squared_error(x_valid, results).numpy()
print("MSE: ", mse)
plt.show()
"""

# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()
#Env.finish_test(session["Number"] + "_" + interpolation.title() + "_" + sync_method.replace(" ", ""))