# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file MultiLayerPerceptron.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date April, 2021
@details Feed-forward Neural Network using MultiLayerPerceptron algorithm in the output layer.
"""

from Yomchi import \
    Environment as Env, \
    preprocessing as data, \
    models, \
    visualization as ui

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
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
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(51)
np.random.seed(51)

# Session, data and methods Parameters
session = 771  # 765 or 771
interpolation = "SLERP"  # or "linear", "quadratic", "cubic", "nearest", "SLERP"
rate_used = data.LFP_DATAMAX_SAMPLING_RATE  # or data.POSITION_DATA_SAMPLING_RATE

# Windowing properties
window_size = 1250  # Equals to 100ms at 1250Hz. Recommended between 100ms to 200ms
batch_size = 32
shuffle_buffer = 1000
lfp_channel = 40
batches_to_plot = 100

# Input pickle file.
i_pickle_file_name = f"S-{session}_C{lfp_channel}_I-{interpolation}_F-{rate_used}_W-{int(window_size*1e3/rate_used)}" \
                     f"ms.pickle"
parameters_dictionary = {"Recording Session Number": str(session),
                        "Interpolation Method": interpolation.title(),
                        "Sampling Rate to Use": str(rate_used) + "Hz",
                        "Window Size": window_size,
                        "Batch size (# of windows)": batch_size,
                        "LFP Signal channel to use": lfp_channel,
                        "Shuffle buffer size": str(shuffle_buffer),
                        "Pickle file used": i_pickle_file_name,
                        "Batches to Plot": batches_to_plot}

Env.print_parameters(parameters_dictionary)

## <li> Step 1
# <ul>
# <li> Import training, validation and test datasets of LFPs as inputs and Angles as labels.
# </ul>
Env.step("Import training, validation and test datasets of LFPs as inputs and Angles as labels.")

with open(f"{data.PATH_TO_DATASETS}Pickles/{i_pickle_file_name}", "rb") as f:
    train_array, valid_array, test_array = pickle.load(f)


## <li> Step 2
# <ul>
# <li> Convert data to windowed series.
# </ul>
Env.step("Convert data to windowed series.")

# The train data is shuffled so the network is trained with a variety of examples.
train_data = data.channels_to_windows(train_array, lfp_channel, window_size, batch_size, shuffle_buffer)

# The validation and testing data are not shuffled because the intention is to predict the series in order.
val_data = data.channels_to_windows(valid_array, lfp_channel, window_size, batch_size)
test_data = data.channels_to_windows(test_array, lfp_channel, window_size, batch_size)

for example_inputs, example_labels in train_data.take(1):
  Env.print_text(f'Inputs shape (batch, time, channels): {example_inputs.shape}')
  Env.print_text(f'Labels shape (batch, time, labels): {example_labels.shape}')

## <li> Step 3
# <ul>
# <li> Create model: Multi Layer Perceptron (MLP)
# </ul>
Env.step("Create model: Multi Layer Perceptron (MLP)")

# Parameters
units_per_layer = 32
dropout = None
# dropout = 0.60
layers = 3
epochs = 2

model = models.MLP(layers, units_per_layer, dropout)
# model = models.LSTM(units_per_layer)
# model = models.CNN(window_size, units_per_layer)

for example_inputs, example_labels in train_data.take(1):
  Env.print_text(f'Input shape: {example_inputs.shape}')
  Env.print_text(f'Output shape: {model(example_inputs).shape}')

## <li> Step 4
# <ul>
# <li> Compile and Fit
# </ul>
Env.step("Compiling and Fitting the model")

history = models.compile_and_fit(model, train_data, val_data, epochs=epochs)

Env.print_text(f"\nTraining metrics history:")
Env.print_text(f"Loss: {history.history['loss']}")
Env.print_text(f"Mean Absolute Error: {history.history['mean_absolute_error']}")
Env.print_text(f"\nValidation metrics history:")
Env.print_text(f"Loss: {history.history['val_loss']}")
Env.print_text(f"Mean Absolute Error: {history.history['val_mean_absolute_error']}")

## <li> Step 5
# <ul>
# <li> Plot
# </ul>
Env.step(f"Plotting the original angular data vs the predictions of {batches_to_plot} batches")

lfps = []
real = []
pred = []
for inputs, label in test_data.take(batches_to_plot):
    # Getting original data
    real = np.append(real, label[:, 0, 0].numpy())
    lfps = np.append(lfps, inputs[:, 0, 0].numpy())
    # Getting predicted data
    pred = np.append(pred, model.predict(inputs))

# Plotting
figname = "predictions"
plt.figure(figname, figsize=(12, 8))
plt.subplot(211)
plt.title(f"Original angles vs predictions")
plt.ylabel('Angles')
plt.scatter(range(len(real)), real, edgecolors='k', label='Originals', c='#2ca02c', s=16)
plt.scatter(range(len(pred)), pred, marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=16)
plt.legend()

plt.subplot(212)
plt.title(f'LFP channel {lfp_channel}')
plt.scatter(range(len(lfps)), lfps, edgecolors='k', label='LFPs', c='#0a7fdb', s=16)
plt.legend()
ui.store_figure(figname, show=Env.debug)

# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()
# Env.finish_test(f"M-MLP_S-{session}_I-{interpolation}_F-{rate_used}")