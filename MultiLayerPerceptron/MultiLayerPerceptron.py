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

import Yomchi.Environment as Env
import Yomchi.preprocessing as data
import Yomchi.models as models
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
tf.random.set_seed(51)
np.random.seed(51)

# Session, data and methods Parameters
session = 765  # or 771
interpolation = "SLERP"  # or "linear", "quadratic", "cubic", "nearest", "SLERP"
rate_used = data.LFP_DATAMAX_SAMPLING_RATE  # or data.POSITION_DATA_SAMPLING_RATE

# Windowing properties
window_size = 125  # Equals to 100ms at 1250Hz. Recommended between 100ms to 200ms
batch_size = 32
shuffle_buffer = 1000
lfp_channel = 70

# Input pickle file.
i_pickle_file_name = f"S-{session}_I-{interpolation}_F-{rate_used}_W-{int(window_size*1e3/rate_used)}ms.pickle"

parameters_dictionary = {"Recording Session Number": str(session),
                        "Interpolation Method": interpolation.title(),
                        "Sampling Rate to Use": str(rate_used) + "Hz",
                        "Window Size": window_size,
                        "Batch size (# of windows)": batch_size,
                        "LFP Signal channel to use": lfp_channel,
                        "Shuffle buffer size": shuffle_buffer,
                        "Pickle file used": i_pickle_file_name}

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

train_data = data.channels_to_windows(train_array, lfp_channel, window_size, batch_size, shuffle_buffer)
val_data = data.channels_to_windows(valid_array, lfp_channel, window_size, batch_size, shuffle_buffer)
test_data = data.channels_to_windows(test_array, lfp_channel, window_size, batch_size, shuffle_buffer)

for example_inputs, example_labels in train_data.take(1):
  Env.print_text(f'Inputs shape (batch, time, samples): {example_inputs.shape}')
  Env.print_text(f'Labels shape (batch, time, labels): {example_labels.shape}')

## <li> Step 5
# <ul>
# <li> Create model: Multi Layer Perceptron (MLP) with:
# </ul>
Env.step()

# Parameters
units_per_layer = 64
dropout = None
# dropout = 0.60
layers = 3
epochs = 2

model = models.MLP(layers, units_per_layer, dropout)

## <li> Step 6
# <ul>
# <li> Compile and Fit
# </ul>
Env.step()

start_time = time.time()
history = models.compile_and_fit(model, train_data, val_data, epochs=epochs)

# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()
#Env.finish_test(session["Number"] + "_" + interpolation.title() + "_" + sync_method.replace(" ", ""))