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
import Yomchi.visualization as ui

import matplotlib.pyplot as plt
import numpy as np

## <h1> Feed-Forward Neural Network </h1>
# <h2> Experiment Setup </h2>
# <ul>
# <li> Initialize Environment
# <li> Define parameters...
# </ul>
# <ol>
Env.init_environment(True)


## <li> Step 1
# <ul>
# <li> Import LFP data.
# <li> Downsample LFP data to match Angles sampling rate.
# </ul>
Env.step("Importing and prepare LFP data.")

W = data.load_lfp_data(data.LFP_765)
W_downsampled = data.downsample_lfps(W, data.lfp_SamplingRate, data.angles_SamplingRate)

## <li> Step 2
# <ul>
# <li> Import angles data.
# <li> Fill angles gaps to match the LFP sampling rate.
# <li> Interpolate original and padded angles data using a 'quadratic' approach.
# </ul>
Env.step("Import and prepare Angles data")

Y = data.load_angles_data(data.ANGLES_765)
Y_filled = data.pad_angles(Y, data.angles_SamplingRate, data.lfp_SamplingRate)
Y_interpolated = data.interpolate_angles(Y)
Y_filled_interpolated = data.interpolate_angles(Y_filled)

## <li> Step 6
# <ul>
# <li> Label data by concatenating LFPs and padded Angles in a single 2D-array
# <li> Label data by concatenating downsampled LFPs and Angles in a single 2D-array
# </ul>
# </ol>
Env.step("Label data by concatenating LFPs and  Angles in a single 2D-array.")

labeled_data_filled = data.add_labels(W, np.expand_dims(Y_filled, axis=1))
labeled_data_downsampled = data.add_labels(W_downsampled, np.expand_dims(Y, axis=1))

