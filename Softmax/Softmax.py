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
Env.init_environment(True)

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
window_size = 31
batch_size = 31
shuffle_buffer = 1000
lfp_channel = 0
round_angles = False
base_angle = 0  # Unused if round_angles = False
offset_between_angles = 30  # Unused if round_angles = False

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
# <li> Convert the data to a windowed series.
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

Env.print_box("INITIALIZING TENSORFLOW")
windowed_data = data.channels_to_windows(clean_dataset, lfp_channel, window_size, batch_size, shuffle_buffer)

# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()
#Env.finish_test(session["Number"] + "_" + interpolation.title() + "_" + sync_method.replace(" ", ""))