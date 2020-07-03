# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file DataExploration.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details Script for explore the input data by manipulating, printing and plotting the information in different ways.
"""

import Yomchi.Environment as Env
import Yomchi.preprocessing as data
import Yomchi.visualization as ui

import matplotlib.pyplot as plt
import numpy as np

## <h1> Data Exploration </h1>
# <h2> Experiment Setup </h2>
# <ul>
# <li> Initialize Environment
# <li> Define parameters...
# </ul>
# <ol>
Env.init_environment(True)

PLOT = True
DONT_PLOT = False

## <li> Step 1
# <ul>
# <li> Import LFP data.
# <li> Plot channels 0 and 97.
# </ul>
Env.step("Importing LFP data.")
lfp_data = data.load_lfp_data(data.LFP_771)

if DONT_PLOT:
    Env.print_text("Plotting LFP data from channels 0 and 97.")
    plt.figure("LFP_Channels_0_and_97", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.subplot(211)
    plt.plot(lfp_data[:, 0], "xr", markersize=1)
    plt.title("Señal LFP del Canal 0")

    plt.subplot(212)
    plt.plot(lfp_data[:, 97], "xb", markersize=1)
    plt.title("Señal LFP del Canal 97")
    ui.store_figure("LFP_Channels_0_and_97", "LFP_C0-C97.png")

## <li> Step 2
# <ul>
# <li> Downsample LFP data to match Angles sampling rate.
# <li> Plot channels 0 and 97 downsampled.
# </ul>
Env.step("Downsample LFP data to match Angles sampling rate.")

lfp_data_downsampled = data.downsample_lfps(lfp_data, data.lfp_SamplingRate, data.angles_SamplingRate)

if DONT_PLOT:
    Env.print_text("Plotting LFP downsampled data to " + str(data.angles_SamplingRate) + "Hz from channels 0 and 97.")
    plt.figure("LFP_downsampled_0_and_97", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.subplot(211)
    plt.plot(lfp_data_downsampled[:, 0], "xr", markersize=1)
    plt.title("Señal LFP del Canal 0 a " + str(data.angles_SamplingRate) + "Hz.")

    plt.subplot(212)
    plt.plot(lfp_data_downsampled[:, 97], "xb", markersize=1)
    plt.title("Señal LFP del Canal 97 a " + str(data.angles_SamplingRate) + "Hz.")
    ui.store_figure("LFP_downsampled_0_and_97", "LFP_downsampled_C0-C97.png")

## <li> Step 3
# <ul>
# <li> Import angles data.
# <li> Plot angles data.
# </ul>
Env.step("Importing angles data")

angles_data = data.load_angles_data(data.ANGLES_771)

if DONT_PLOT:
    Env.print_text("Plotting Angles data [°]")
    plt.figure("Angles_degrees", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.plot(angles_data[:], "xr", markersize=1)
    plt.title("Información de ángulos [°]")
    ui.store_figure("Angles_degrees", "angles_degrees.png")

## <li> Step 4
# <ul>
# <li> Fill angles gaps to match the LFP sampling rate.
# <li> Plot angles data in the new sampling rate.
# </ul>
Env.step("Pad angles data to reach a higher sampling rate")

angles_data_filled = data.pad_angles(angles_data, data.angles_SamplingRate, data.lfp_SamplingRate)

if DONT_PLOT:
    Env.print_text("Plotting Angles data padded with 'NaN' [°]")
    plt.figure("Angles_padded_degrees", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.plot(angles_data_filled[:], "xb", markersize=1)
    plt.title("Información de ángulos a " + str(data.lfp_SamplingRate) + "Hz. [°]")
    ui.store_figure("Angles_padded_degrees", "angles_padded_degrees.png")

## <li> Step 5
# <ul>
# <ul>
# <li> Interpolate original and padded angles data using a 'quadratic' approach.
# <li> Plot original and padded angles data after quadratic interpolation.
# </ul>
Env.step("Interpolate original and padded angles data using a 'quadratic' approach.")

angles_data_interpolated = data.interpolate_angles(angles_data, "quadratic")
angles_data_filled_interpolated = data.interpolate_angles(angles_data_filled, "quadratic")

if DONT_PLOT:
    Env.print_text("Plotting original Angles data after interpolation [°]")
    plt.figure("Angles_inter_degrees", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.plot(angles_data_interpolated[:], "xr", markersize=1)
    plt.title("Información de ángulos interpolada. [°]")
    ui.store_figure("Angles_inter_degrees", "angles_inter_degrees.png")

    Env.print_text("Plotting padded Angles data after interpolation [°]")
    plt.figure("Angles_padded_inter_degrees", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.plot(angles_data_filled_interpolated[:], "xb", markersize=1)
    plt.title("Información de ángulos a " + str(data.lfp_SamplingRate) + "Hz interpolada. [°]")
    ui.store_figure("Angles_padded_inter_degrees", "angles_padded_inter_degrees.png")

## <li> Step 6
# <ul>
# <li> Label data by concatenating LFPs and padded, interpolated Angles in a single 2D-array.
# <li> Plot LFP and angles data.
# </ul>
Env.step("Label data by concatenating LFPs and padded, interpolated Angles in a single 2D-array.")

labeled_data_filled = data.add_labels(lfp_data, np.expand_dims(angles_data_filled_interpolated, axis=1))

if DONT_PLOT:
    Env.print_text("Plotting LFP data from channels 0 and padded, interpolated angles data at "
                   + str(data.lfp_SamplingRate) + "Hz. [°]")

    plt.figure("LFP_C0_and_angles_inter_padded_39Hz", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.subplot(211)
    plt.plot(labeled_data_filled[:-1, 0], "xr", markersize=1)
    plt.title("Señal LFP del Canal 0 muestreada a " + str(data.lfp_SamplingRate) + "Hz")

    plt.subplot(212)
    plt.plot(labeled_data_filled[:, -1], "xb", markersize=1)
    plt.title("Información de ángulos a " + str(data.lfp_SamplingRate) + "Hz interpolada y rellena. [°]")
    ui.store_figure("LFP_C0_and_angles_inter_padded_39Hz", "LFP_C0_and_angles_inter_padded_39Hz.png")

## <li> Step 7
# <ul>
# <li> Label data by concatenating downsampled LFPs and interpolated Angles in a single 2D-array.
# <li> Plot LFP and angles data.
# </ul>
Env.step("Label data by concatenating downsampled LFPs and interpolated Angles in a single 2D-array.")

labeled_data_downsampled = data.add_labels(lfp_data_downsampled, np.expand_dims(angles_data_interpolated, axis=1))

# print(angles_data[15560:15600])
# print(labeled_data_downsampled[15560:15600, -1])

if DONT_PLOT:
    Env.print_text("Plotting downsampled LFP data from channels 0 and interpolated angles data at "
                   + str(data.angles_SamplingRate) + "Hz. [°]")

    plt.figure("LFP_C0_down_and_angles_inter_39Hz", figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.subplot(211)
    plt.plot(labeled_data_downsampled[:-1, 0], "xr", markersize=1)
    plt.title("Señal LFP del Canal 0 muestreada a " + str(data.angles_SamplingRate) + "Hz")

    plt.subplot(212)
    plt.plot(labeled_data_downsampled[:, -1], "xb", markersize=1)
    plt.title("Información de ángulos a " + str(data.angles_SamplingRate) + "Hz interpolada. [°]")
    ui.store_figure("LFP_C0_down_and_angles_inter_39Hz", "LFP_C0_down_and_angles_inter_39Hz.png")

# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()