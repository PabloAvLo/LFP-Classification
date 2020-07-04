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

## <h1> Data Exploration </h1>
# <h2> Experiment Setup </h2>
# <ul>
# <li> Initialize Environment
# <li> Specify run configuration parameters.
# <li> Specify session and methods parameters.
# <li> Specify data properties parameters.
# </ul>
# <ol>
Env.init_environment(True)

# Run configuration parameters
PLOT = True
DONT_PLOT = False

# Session and methods Parameters
# session = {"Number": "771", "LFP Data": data.LFP_771, "Angles Data": data.ANGLES_771}  # or
session = {"Number": "765", "LFP Data": data.LFP_765, "Angles Data": data.ANGLES_765}

interpolation = "linear"  # "quadratic"
sync_method = "Downsample LFPs"  # "Pad angles"

# Data Properties
num_channels = data.EC014_41_NUMBER_OF_CHANNELS
rate_used = data.POSITION_DATA_SAMPLING_RATE
if sync_method == "Pad angles":
    rate_used = data.LFP_DATAMAX_SAMPLING_RATE

Env.print_parameters({"Recording Session Number": session["Number"],
                      "Interpolation Method": interpolation.title(),
                      "Synchronization Method": sync_method,
                      "Number of Recorded Channels": str(num_channels),
                      "Sampling Rate to Use": str(rate_used) + "Hz"})

## <li> Step 1
# <ul>
# <li> Import LFP data.
# <li> Plot channels 0 and 97.
# </ul>
Env.step("Importing LFP data from session: " + session["Number"])
lfp_data = data.load_lfp_data(session["LFP Data"])

if PLOT:
    Env.print_text("Plotting LFP data from channels 0 and 97 at " + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz")
    figname = session["Number"] + "_LFP_C0_and_C97_" + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz"
    plt.figure(figname, figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.subplot(211)
    plt.plot(lfp_data[:, 0], "xr", markersize=1)
    plt.title("Señal LFP. Sesión: " + session["Number"] + " del Canal 0 a "
              + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz")

    plt.subplot(212)
    plt.plot(lfp_data[:, 97], "xb", markersize=1)
    plt.title("Señal LFP. Sesión: " + session["Number"] + " del Canal 97 a "
              + str(data.LFP_DATAMAX_SAMPLING_RATE) + "Hz")
    ui.store_figure(figname)

## <li> Step 2
# <ul>
# <li> Import angles data.
# <li> Plot angles data.
# </ul>
Env.step("Importing angles data from session: " + session["Number"])

angles_data = data.load_angles_data(session["Angles Data"])

if PLOT:
    Env.print_text("Plotting Angles data [°] at " + str(data.POSITION_DATA_SAMPLING_RATE) + "Hz")
    figname = session["Number"] + "_Angles_degrees_" + str(data.POSITION_DATA_SAMPLING_RATE) + "Hz"
    plt.figure(figname, figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.plot(angles_data[:], "xr", markersize=1)
    plt.title("Información de ángulos [°]. Sesión: " + session["Number"] + " a "
              + str(data.POSITION_DATA_SAMPLING_RATE) + "Hz")
    ui.store_figure(figname)

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
        figname = session["Number"] + "_LFP_downsampled_0_and_97_" + str(rate_used) + "Hz"
        plt.figure(figname, figsize=ui.FIG_SIZE, dpi=ui.DPI)
        plt.subplot(211)
        plt.plot(lfp_data[:, 0], "xr", markersize=1)
        plt.title("Señal LFP del Canal 0 a " + str(rate_used) + "Hz. Sesión: " + session["Number"])

        plt.subplot(212)
        plt.plot(lfp_data[:, 97], "xb", markersize=1)
        plt.title("Señal LFP del Canal 97 a " + str(rate_used) + "Hz. Sesión: " + session["Number"])
        ui.store_figure(figname)

elif sync_method == "Pad angles":
    ## <li> Step 3
    # <ul>
    # <li> Fill angles gaps to match the LFP sampling rate.
    # <li> Plot angles data in the new sampling rate.
    # </ul>
    Env.step("Pad angles data to reach a higher sampling rate")

    angles_data = data.pad_angles(angles_data, data.POSITION_DATA_SAMPLING_RATE, data.LFP_DATAMAX_SAMPLING_RATE)

    if PLOT:
        Env.print_text("Plotting Angles data padded with 'NaN' [°]")
        figname = session["Number"] + "_Angles_padded_degrees_" + str(rate_used) + "Hz"
        plt.figure(figname, figsize=ui.FIG_SIZE, dpi=ui.DPI)
        plt.plot(angles_data[:], "xb", markersize=1)
        plt.title("Información de ángulos [°] rellena a " + str(rate_used) + "Hz. Sesión: " + session["Number"])
        ui.store_figure(figname)

## <li> Step 4
# <ul>
# <li> Interpolate angles data using a 'interpolation' approach.
# <li> Plot angles data after interpolation.
# </ul>
Env.step("Interpolate angles data using a " + interpolation + " approach.", 4)

angles_data_interpolated = data.interpolate_angles(angles_data, interpolation)

if PLOT:
    Env.print_text("Plotting Angles data after " + interpolation + " interpolation [°]")
    figname = session["Number"] + "_Angles_" + interpolation + "_degrees_" + str(rate_used) + "Hz"
    plt.figure(figname, figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.plot(angles_data_interpolated[:], "xr", markersize=1)
    plt.title("Información de ángulos [°]. Sesión: " + session["Number"] + ". Interpolada con: " + interpolation
              + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname)

## <li> Step 5
# <ul>
# <li> Label data by concatenating LFPs and interpolated Angles in a single 2D-array.
# <li> Plot LFP and angles data.
# <li> Print an angles data window where the first LED is unsynchronized and then the second LED is lost too.
# </ul>
Env.step("Label data by concatenating LFPs and interpolated Angles in a single 2D-array.")

labeled_data = data.add_labels(lfp_data, np.expand_dims(angles_data_interpolated, axis=1))

if PLOT:
    Env.print_text("Plotting LFP data from channels 0 and interpolated angles data at " + str(rate_used) + "Hz. [°]")
    figname = session["Number"] + "_LFP_C0_and_angles_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname, figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.subplot(211)
    plt.plot(labeled_data[:-1, 0], "xr", markersize=1)
    plt.title("Señal LFP del Canal 0. Muestreada a " + str(rate_used) + "Hz. Sesión: " + session["Number"])

    plt.subplot(212)
    plt.plot(labeled_data[:, -1], "xb", markersize=1)
    plt.title("Información de ángulos [°]. Sesión: " + session["Number"] + ". Interpolada con: " + interpolation
              + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname)

if session["Number"] == "771" and sync_method == "Downsample LFPs":
    Env.print_text("\nAngles Data from: 15560 to 15600 where at 15566 the first LED is lost and at 15583 both are lost")
    Env.print_text(' '.join([str(elem) for elem in angles_data[15560:15600]]))

    Env.print_text("\nAngles Data Interpolated from: 15560 to 15600 where at 15566 the first LED is lost and at 15583 "
                   "both are lost.")
    Env.print_text(' '.join([str(elem) for elem in labeled_data[15560:15600, -1]]))

## <li> Step 6
# <ul>
# <li> Clean the labeled dataset from NaN values at the boundaries.
# <li> Print some NaN counts in different dataset stages.
# <li> Plot LFP and angles data already clean.
# </ul>
Env.step("Clean the labeled dataset from NaN values at the boundaries.")

Env.print_text("Number of NaNs in Angles Data without interpolation: " + str(np.count_nonzero(np.isnan(angles_data))))
Env.print_text("Number of NaNs in Labeled Dataset with interpolated Angles Data: "
               + str(np.count_nonzero(np.isnan(labeled_data))))

nans_begin = 0
nans_end = 0
length = len(labeled_data)
for i in range(length):
    if ~np.isnan(labeled_data[i, -1]):
        nans_begin = i
        break
Env.print_text("Number of NaNs at the beginning of the interpolated Angles Data: " + str(nans_begin))

for i in range(length-1, 0, -1):
    if ~np.isnan(labeled_data[i, -1]):
        nans_end = i
        break

nans_end = length - (nans_end + 1)
Env.print_text("Number of NaNs at the end of the interpolated Angles Data: " + str(nans_end))

clean_dataset = data.clean_unsync_boundaries(labeled_data)
Env.print_text("Number of NaNs in Labeled and Clean Dataset with interpolated Angles Data: "
               + str(np.count_nonzero(np.isnan(clean_dataset))))

if PLOT:
    Env.print_text("Plotting clean LFP data from channels 0 and interpolated angles data at "
                   + str(rate_used) + "Hz. [°]")
    figname = session["Number"] + "_LFP_C0_clean_and_angles_" + interpolation + "_" + str(rate_used) + "Hz"
    plt.figure(figname, figsize=ui.FIG_SIZE, dpi=ui.DPI)
    plt.subplot(211)
    plt.plot(labeled_data[:-1, 0], "xr", markersize=1)
    plt.title("Señal LFP del Canal 0 limpia. Muestreada a " + str(rate_used) + "Hz. Sesión: " + session["Number"])
    plt.subplot(212)
    plt.plot(labeled_data[:, -1], "xb", markersize=1)
    plt.title("Información de ángulos [°] limpia. Sesión: " + session["Number"] + ". Interpolada con: "
              + interpolation + " a " + str(rate_used) + "Hz")
    ui.store_figure(figname)


## <li> Step 7
# <ul>
# <li> Discretize the angles into bins
# <li> Box plot the angles data
# </ul>

angles = labeled_data[:, -1]
a = angles[(-180 <= angles) & (angles< -150)]
angles_separated = {
                    "-180 to -150": angles[(-180 <= angles) & (angles< -150)],
                    "-150 to -120": angles[(-150 <= angles) & (angles< -120)],
                    "-120 to -90" : angles[(-120 <= angles) & (angles< -90)],
                    "-90 to -60"  : angles[(-90  <= angles) & (angles< -60)],
                    "-60 to -30"  : angles[(-60  <= angles) & (angles< -30)],
                    "-30 to 0"    : angles[(-30  <= angles) & (angles< 0)],
                    "0 to 30"     : angles[(0    <= angles) & (angles< 30)],
                    "30 to 60"    : angles[(30   <= angles) & (angles< 60)],
                    "60 to 90"    : angles[(60   <= angles) & (angles< 90)],
                    "90 to 120"   : angles[(90   <= angles) & (angles< 120)],
                    "120 to 150"  : angles[(120  <= angles) & (angles< 150)],
                    "150 to 180"  : angles[(150  <= angles) & (angles<= 180)]}

df = pd.DataFrame(angles_separated, columns=["-150 to -120", "-120 to -90", "-90 to -60", "-60 to -30", "-30 to 0",
                                             "0 to 30", "30 to 60", "60 to 90", "90 to 120", "120 to 150", "150 to 180"])

for col in df.select_dtypes(include='object').columns:
    pd.value_counts(df[col]).plot.barh()
    plt.title(col)
    plt.show()


# </ol>
## <h2> Finish Test and Exit </h2>
Env.finish_test()
# Env.finish_test(session["Number"] + "_" + interpolation.title() + "_" + sync_method.replace(" ", ""))
