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
import seaborn as sns
import pandas as pd
import tensorflow as tf


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
Env.init_environment(True, enable_debug=True)

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

# Run configuration parameters
PLOT = True

# Session and methods Parameters
session = 771  # or 765

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
# <li> Plot channels 0 and 97.
# </ul>
Env.step(f"Importing LFP data from session: {session}")
lfp_data = data.load_lfp_data(data.LFP[session])

if PLOT:
    Env.print_text("Plotting LFP data from channels 0 and 97 at " + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz")
    figname = str(session) + "_LFP_C0_and_C97_" + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz"
    plt.figure(figname)
    plt.subplot(211)
    plt.plot(lfp_data[:, 0], "xr")
    plt.title("Señal LFP. Sesión: " + str(session) + " del Canal 0 a "
              + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz")

    plt.subplot(212)
    plt.plot(lfp_data[:, 97], "xb")
    plt.title("Señal LFP. Sesión: " + str(session) + " del Canal 97 a "
              + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz")
    ui.store_figure(figname, True)

## <li> Step 2
# <ul>
# <li> Import angles data.
# <li> Plot angles data.
# </ul>
Env.step(f"Importing angles data from session: {session}")

angles_data = data.load_angles_data(data.ANGLES[session])

Env.print_text("Min angle: " + str(np.nanmin(angles_data)) + ", Max angle: " + str(np.nanmax(angles_data)))

if PLOT:
    Env.print_text("Plotting Angles data [°] at " + str(data.POSITION_DATA_SAMPLING_RATE) + "Hz")
    figname = str(session) + "_Angles_degrees_" + str(data.POSITION_DATA_SAMPLING_RATE) + "Hz"
    plt.figure(figname)
    plt.plot(angles_data[:], "xr")
    plt.title("Información de ángulos [°]. Sesión: " + str(session) + " a "
              + str(data.POSITION_DATA_SAMPLING_RATE) + "Hz")
    ui.store_figure(figname, True)

if sync_method == "Downsample LFPs":
    ## <li> Step 3
    # <ul>
    # <li> Downsample LFP data to match Angles sampling rate.
    # <li> Plot channels 0 and 97 downsampled.
    # </ul>
    Env.step("Downsample LFP data to match Angles sampling rate.")

    lfp_data = data.downsample_lfps(lfp_data, data.LFP_DATAMAX_SAMPLING_RATE, data.POSITION_DATA_SAMPLING_RATE)

    if PLOT:
        Env.print_text("Plotting LFP downsampled data to " + str(rate_used) + "Hz from channels 0 and 97.")
        figname = str(session) + "_LFP_downsampled_0_and_97_" + str(rate_used) + "Hz"
        plt.figure(figname)
        plt.subplot(211)
        plt.plot(lfp_data[:, 0], "xr")
        plt.title("Señal LFP del Canal 0 a " + str(rate_used) + "Hz. Sesión: " + str(session))

        plt.subplot(212)
        plt.plot(lfp_data[:, 97], "xb")
        plt.title("Señal LFP del Canal 97 a " + str(rate_used) + "Hz. Sesión: " + str(session))
        ui.store_figure(figname, True)

elif sync_method == "Upsample Angles":
    ## <li> Step 3
    # <ul>
    # <li> Fill angles gaps to match the LFP sampling rate.
    # <li> Plot angles data in the new sampling rate.
    # </ul>
    Env.step("Upsample Angles data to reach a higher sampling rate")

    angles_data = data.angles_expansion(angles_data, data.POSITION_DATA_SAMPLING_RATE, data.LFP_DATAMAX_SAMPLING_RATE)

    if PLOT:
        Env.print_text("Plotting Angles data expanded with 'NaN' [°]")
        figname = str(session) + "Expanded_Angles_degrees_" + str(rate_used) + "Hz"
        plt.figure(figname)
        plt.plot(angles_data[:], "xb")
        plt.title("Información de ángulos [°] expandida a " + str(rate_used) + "Hz. Sesión: " + str(session))
        ui.store_figure(figname, True)

## <li> Step 4
# <ul>
# <li> Interpolate angles data using a 'interpolation' approach.
# <li> Plot angles data after interpolation.
# </ul>
Env.step("Interpolate angles data using a " + interpolation + " approach.", 4)

angles_data_interpolated = data.interpolate_angles(angles_data, interpolation)

if PLOT:
    Env.print_text("Plotting Angles data after " + interpolation + " interpolation [°]")
    figname = str(session) + "_Angles_" + interpolation + "_degrees_" + str(rate_used) + "Hz"
    plt.figure(figname)
    plt.plot(angles_data_interpolated[:], "xr")
    plt.title("Información de ángulos [°]. Sesión: " + str(session) + ".\n Interpolada con: " + interpolation
              + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname, True)

## <li> Step 5
# <ul>
# <li> Label data by concatenating LFPs and interpolated Angles in a single 2D-array.
# <li> Plot LFP and angles data.
# <li> Print an angles data window where the first LED is unsynchronized and then the second LED is lost too.
# </ul>
Env.step("Label data by concatenating LFPs and interpolated Angles in a single 2D-array.")

# IF round_angles: Rounding the angles to be descrete labels, starting from 'base_angle'until 360 on steps of
# 'offset_between_angles'. Else Not rounding the angles to be discrete labels
labeled_data = data.add_labels(lfp_data, np.expand_dims(angles_data_interpolated, axis=1), round_angles,
                               base_angle, offset_between_angles)

if PLOT:
    Env.print_text("Plotting LFP data from channels 0 and interpolated angles data at " + str(rate_used) + "Hz. [°]")
    figname = str(session) + "_LFP_C0_and_angles_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname)
    plt.subplot(211)
    plt.plot(labeled_data[:-1, 0], "xr")
    plt.title("Señal LFP del Canal 0. Muestreada a " + str(rate_used) + "Hz. Sesión: " + str(session))

    plt.subplot(212)
    plt.plot(labeled_data[:, -1], "xb")
    plt.title("Información de ángulos [°]. Sesión: " + str(session) + ".\n Interpolada con: " + interpolation
              + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname, True)

if str(session) == "771" and sync_method == "Downsample LFPs":
    Env.print_text("\nAngles Data from: 15560 to 15600 where at 15566 the first LED is lost and at 15583 both are lost")
    Env.print_text(' '.join([str(elem) for elem in angles_data[15560:15600]]))

    Env.print_text("\nAngles Data Interpolated from: 15560 to 15600 where at 15566 the first LED is lost and at 15583 "
                   "both are lost.")
    Env.print_text(' '.join([str(elem) for elem in labeled_data[15560:15600, -1]]))

## <li> Step 6
# <ul>
# <li> Convert labeled dataset to a dataframe.
# <li> Clean both, the labeled dataset and dataframe from NaN values at the boundaries.
# <li> Print Bar plot for labeled angles.
# </ul>
Env.step("Clean the labeled dataset from NaN values at the boundaries.")

dataframe = data.ndarray_to_dataframe(labeled_data, rate_used)

clean_frame = data.clean_unsync_boundaries(dataframe)
clean_dataset = data.clean_unsync_boundaries(labeled_data, False)

if round_angles:
    labels = np.arange(base_angle, 360, offset_between_angles)
    percentages = []
    for u in range(base_angle, 360, offset_between_angles):
        percentages.append(round(np.sum(clean_dataset[:, -1] == u)*100/len(clean_dataset[:, -1])))

    labels_percent = np.concatenate((np.expand_dims(labels, axis=1), np.expand_dims(percentages, axis=1)), axis=1)
    dataframe_labels = pd.DataFrame(data=labels_percent, columns=["Angulos", "Porcentaje"])

    if PLOT:
        Env.print_text("Plotting Barplot of Labels after " + interpolation + " interpolation")
        figname = str(session) + "_BarPLotAngles_" + interpolation + "_" + str(rate_used) + "Hz"
        plt.figure(figname)
        sns.barplot(x="Angulos", y="Porcentaje", data=dataframe_labels)
        plt.title("Gráfico de Barras de las etiquetas. Sesión: " + str(session) + ".\n Interpolada con: " +
                  interpolation + " a " + str(rate_used) + "Hz")
        ui.store_figure(figname, True)

## <li> Step 7
# <ul>
# <li> Count number of NaNs at the beginning and at the end of the data.
# <li> Print some NaN counts in different dataset stages.
# </ul>
Env.step("Count and print number of NaNs in the Dataset.")

nans_begin = 0
nans_end = 0
length = len(angles_data_interpolated)
for i in range(length):
    if ~np.isnan(angles_data_interpolated[i]):
        nans_begin = i
        break
for i in range(length-1, 0, -1):
    if ~np.isnan(angles_data_interpolated[i]):
        nans_end = i
        break
nans_end = length - (nans_end + 1)

Env.print_text("Number of NaNs in Angles Data without interpolation: " + str(np.count_nonzero(np.isnan(angles_data))))
Env.print_text("Number of NaNs in Labeled Dataset with interpolated Angles Data: "
               + str(np.count_nonzero(np.isnan(angles_data_interpolated))))
Env.print_text("Number of NaNs at the beginning of the interpolated Angles Data: " + str(nans_begin))
Env.print_text("Number of NaNs at the end of the interpolated Angles Data: " + str(nans_end))
Env.print_text("Number of NaNs in Labeled and Clean Dataset with interpolated Angles Data: "
               + str(np.count_nonzero(np.isnan(clean_dataset))))
Env.print_text("Number of NaNs in Labeled and Clean DataFRAME with interpolated Angles Data: "
               + str(np.count_nonzero(np.isnan(clean_frame))))

if PLOT:
    ## <li> Step 8
    # <ul>
    # <li> Plotting clean LFP data from channels 0 and interpolated angles data.
    # </ul>
    Env.step("Plotting clean LFP data from channels 0 and interpolated angles data at " + str(rate_used) + "Hz. [°]")

    figname = str(session) + "_LFP_C0_clean_and_angles_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname)
    plt.subplot(211)
    plt.plot(labeled_data[:-1, 0], "xr")
    plt.title("Señal LFP del Canal 0 limpia. Muestreada a " + str(rate_used) + "Hz. Sesión: " + str(session))
    plt.subplot(212)
    plt.plot(labeled_data[:, -1], "xb")
    plt.title("Información de ángulos [°] limpia. Sesión: " + str(session) + ".\n Interpolada con: "
              + interpolation + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname, True)

    ## <li> Step 9
    # <ul>
    # <li> Plotting Boxplot of all channels in two figures, from 0-49 and from 50-98.
    # </ul>
    Env.step("Plotting Boxplot of all channels in two figures, from 0-49 and from 50-98.")

    figname = str(session) + "_BoxPLot0-49_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname, figsize=[12, 16])
    sns.boxplot(data=clean_frame.iloc[:, 0:50], orient="h")
    plt.ylabel("Canales")
    plt.xlabel("Voltaje")
    plt.title("Diagrama de caja de los canales 0-49.\nSesión: " + str(session) + ". Interpolada con: "
              + interpolation + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname, True)

    figname = str(session) + "_BoxPLot50-98_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname, figsize=[12, 16])
    sns.boxplot(data=clean_frame.iloc[:, 50:99], orient="h")
    plt.ylabel("Canales")
    plt.xlabel("Voltaje")
    plt.title("Diagrama de caja de los canales 50-98.\nSesión: " + str(session) + ". Interpolada con: "
              + interpolation + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname, True)

    figname = str(session) + "_BoxPLotChannels_" + interpolation + "_" + str(rate_used) + "Hz"
    fig = plt.figure(figname)

    ## <li> Step 10
    # <ul>
    # <li> Plotting Boxplot of interpolated angles.
    # <li> Plotting Boxplot of interpolated angles vs channel 0
    # </ul>
    Env.step("Plotting Boxplot of interpolated angles.")

    figname = str(session) + "_BoxPLotAngles_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname)
    sns.boxplot(x=clean_frame["Angle"])
    plt.ylabel("Muestras")
    plt.xlabel("Ángulos")
    plt.title("Diagrama de caja de los ángulos.\nSesión: " + str(session) + ". Interpolada con: "
              + interpolation + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname, True)

    """ NOT USEFUL INFORMATION. ALSO CRASHES THE EXECUTION
    figname = str(session) + "_BoxPLotC0vsAngles_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname)
    sns.boxplot(x=clean_frame["E1"], y=clean_frame["Angle"])
    plt.title("Diagrama de caja del canal 0 contra los ángulos.\nSesión: " + str(session) + ". Interpolada con: "
              + interpolation + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname, True, True)
    """

## <li> Step 11
    # <ul>
    # <li> Convert data to windowed series.
    # </ul>
Env.step("Convert data to windowed series.", 11)
windowed_data = data.channels_to_windows(clean_dataset, lfp_channel, window_size, batch_size, shuffle_buffer)

# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()
#Env.finish_test(str(session) + "_" + interpolation.title() + "_" + sync_method.replace(" ", ""))
