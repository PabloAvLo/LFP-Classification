# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file DatasetGenerator.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date Mar, 2021
@details Properly loads the input data and labels and prepare a clean dataset.
"""

from Yomchi import \
    Environment as Env, \
    preprocessing as data, \
    visualization as ui

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import pickle

## <h1> Data Exploration </h1>
# <h2> Experiment Setup </h2>
# <ul>
# <li> Initialize Environment.
# <li> Initialize Tensorflow session.
# <li> Set seed for Numpy and Tensorflow
# <li> Specify run configuration parameters.
# <li> Specify session and methods parameters.
# <li> Specify data properties parameters.
# </ul>
# <ol>
Env.init_environment(True)

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

# Session and methods Parameters
# The recording session 771 has a 23.81% of invalid positions, while the session 765 has only 6.98%
session = 771  # or 765
interpolation = "SLERP"  # "linear" "quadratic" "cubic" "nearest" "SLERP"
sync_method = "Upsample Angles"  # "Upsample Angles" "Downsample LFPs"

# Data Properties
num_channels = data.EC014_41_NUMBER_OF_CHANNELS
rate_used = data.POSITION_DATA_SAMPLING_RATE
if sync_method == "Upsample Angles":
    rate_used = data.LFP_DATAMAX_SAMPLING_RATE

# Windowing properties
window_size = 1250  # Equals to 100ms at 1250Hz. Recommended between 100ms to 200ms
batch_size = 32
shuffle_buffer = 1000
lfp_channel = 70
round_angles = False
base_angle = 0  # Unused if round_angles = False
offset_between_angles = 30  # Unused if round_angles = False

extra = {"Round angles to get discrete label": round_angles}
if round_angles:
    extra.update({"Angle labels starting from": str(base_angle) + "°",
                  "until 360° in steps of": str(offset_between_angles) + "°"})

o_pickle_file_name = f"S-{session}_I-{interpolation}_F-{rate_used}_W-{int(window_size*1e3/rate_used)}ms.pickle"

parameters_dictionary = {"Recording Session Number": str(session),
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
# </ul>
Env.step(f"Importing LFP data from session: {session}")
lfp_data = data.load_lfp_data(data.LFP[session])

## <li> Step 2
# <ul>
# <li> Import angles data.
# </ul>
Env.step(f"Importing angles data from session: {session}")
angles_data = data.load_angles_data(data.ANGLES[session])

if sync_method == "Downsample LFPs":
    ## <li> Step 3
    # <ul>
    # <li> Downsample LFP data to match Angles sampling rate.
    # </ul>
    Env.step("Downsample LFP data to match Angles sampling rate.")

    lfp_data = data.downsample_lfps(lfp_data, data.LFP_DATAMAX_SAMPLING_RATE, data.POSITION_DATA_SAMPLING_RATE)

elif sync_method == "Upsample Angles":
    ## <li> Step 3
    # <ul>
    # <li> Fill angles gaps to match the LFP sampling rate.
    # </ul>
    Env.step("Upsample Angles data to reach a higher sampling rate")

    angles_data = data.angles_expansion(angles_data, data.POSITION_DATA_SAMPLING_RATE, data.LFP_DATAMAX_SAMPLING_RATE)


## <li> Step 4
# <ul>
# <li> Label data by concatenating LFPs and Angles in a single 2D-array.
# </ul>
Env.step("Label data by concatenating LFPs and interpolated Angles in a single 2D-array.")

# IF round_angles: Rounding the angles to be discrete labels, starting from 'base_angle' until 360 on steps of
# 'offset_between_angles'. Else, the angles are not rounded to discrete labels.
labeled_data = data.add_labels(lfp_data, np.expand_dims(angles_data, axis=1), round_angles,
                               base_angle, offset_between_angles)


## <li> Step 5
# <ul>
# <li> Clean the labeled dataframe from -1 values, which represent the wrongly acquired positional samples.
# </ul>
Env.step("Clean the labeled dataset from discontinuities in positional data (angles).")

clean_datasets = data.clean_invalid_positional(labeled_data)


## <li> Step 6
# <ul>
# <li> Interpolate angles data using a 'interpolation' approach.
# </ul>
Env.step(f"Interpolate angles data using a {interpolation} approach.")

# Get all LFP channels, excluding the angles.
clean_interpolated_data = [s[:, :-1] for s in clean_datasets]

# Interpolate angles and add them to the dataset
clean_interpolated_angles = [data.interpolate_angles(s[:, -1], interpolation) for s in clean_datasets]

for i in range(len(clean_interpolated_data)):
    clean_interpolated_data[i] = \
        np.concatenate((clean_interpolated_data[i], np.expand_dims(clean_interpolated_angles[i], axis=1)), axis=1)


## <li> Step 7
# <ul>
# <li> Plotting Angles and one channel of the LFPs after cleaning and interpolation.
# </ul>
Env.step("Plotting Angles and one channel of the LFPs after cleaning and interpolation.")

Env.print_text(f"Plotting Angles data [°] at {rate_used}Hz")
figname = f"{session}_Angles_degrees_{rate_used}Hz"
plt.figure(figname)
plt.plot(clean_interpolated_data[4][:, -1], "xr")
plt.title(f"Información de ángulos [°]. Sesión: {session} a {rate_used}Hz")
ui.store_figure(figname, show=Env.debug)

Env.print_text(f"Plotting LFP Channel {lfp_channel} at {rate_used}Hz")

figname = f"{session}_LFP_c{lfp_channel}_{rate_used}Hz"
plt.figure(figname)
plt.plot(clean_interpolated_data[4][:, lfp_channel], "x-r")
plt.title(f"Información de LFPs Canal {lfp_channel}. Sesión: {session} a {rate_used}Hz")
ui.store_figure(figname, show=Env.debug)

## <li> Step 8
# <ul>
# <li> Get preferred angle according to LFPs channel.
# <li> Plotting the sum of LFPs per angle from 0 to 359 degrees.
# </ul>
Env.step("Get Preferred angle.")

angles = [0] * 361

for subset in clean_interpolated_data:
    for index in range(len(subset)):
        angles[int(round(subset[index, -1]))] += abs(subset[index, lfp_channel])

angles[0] += angles[360]
angles = angles[:-1]
preferred_angle = angles.index(max(angles))

Env.print_text(f" Preferred angle according to LFP Channel {lfp_channel}: {preferred_angle}°")

Env.print_text(f"Plotting Preferred angle according to LFP Channel {lfp_channel}")
figname = f"{session}_preferred_angle_LFP_c{lfp_channel}_{rate_used}Hz"
plt.figure(figname)
plt.plot(range(360), angles, "x-r")
plt.title(f"Potencia de LFPs en el Canal {lfp_channel} por angulo. Sesión: {session} a {rate_used}Hz")
ui.store_figure(figname, show=Env.debug)

## <li> Step 9
# <ul>
# <li> Get largest subset of data.
# </ul>
Env.step("Get largest subset of data.")

max_length = 0
max_subset_index = 0
for index, subset in enumerate(clean_interpolated_data):
    if max_length < len(subset):
        max_length = len(subset)
        max_subset_index = index

largest_subset = clean_interpolated_data[max_subset_index]
Env.print_text(f"Sahpe of the largest subset of data : [{len(largest_subset)},{len(largest_subset[0])}]")

## <li> Step 10
# <ul>
# <li> Split the data in training set and validation set.
# </ul>
Env.step()

n = len(largest_subset)
train_array = largest_subset[0:int(n*0.7), :]
valid_array = largest_subset[int(n*0.7):int(n*0.9), :]
test_array = largest_subset[int(n*0.9):, :]

Env.print_text(f"Training data shape: [{len(train_array)},{len(train_array[0])}]")
Env.print_text(f"Validation data shape: [{len(valid_array)},{len(valid_array[0])}]")
Env.print_text(f"Test data shape: [{len(test_array)},{len(test_array[0])}]")

## <li> Step 11
# <ul>
# <li> Save the dataset to a pickle file
# </ul>
Env.step(f"Save the dataset to pickle file: {o_pickle_file_name}.")

with open(f"{Env.RESULTS_FOLDER}/{o_pickle_file_name}", 'wb') as f:
    pickle.dump([train_array, valid_array, test_array], f)


## <li> Step 12
# <ul>
# <li> Convert data to windowed series.
# </ul>
Env.step("Convert data to windowed series.")

train_data = data.channels_to_windows(train_array, lfp_channel, window_size, batch_size, shuffle_buffer)
val_data = data.channels_to_windows(valid_array, lfp_channel, window_size, batch_size, shuffle_buffer)
test_data = data.channels_to_windows(test_array, lfp_channel, window_size, batch_size, shuffle_buffer)

# TODO: The shape should be (batch, time, features) to be compatible with what tensorflow expects as default.
for example_inputs, example_labels in train_data.take(1):
  Env.print_text(f'Inputs shape (batch, time, samples): {example_inputs.shape}')
  Env.print_text(f'Labels shape (batch, time, labels): {example_labels.shape}')

# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()
#Env.finish_test(session["Number"] + "_" + interpolation.title() + "_" + sync_method.replace(" ", ""))
