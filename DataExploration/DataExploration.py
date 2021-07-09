# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file DataExploration.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date July, 2021
@details Script for explore the input data by manipulating, printing and Plot the information in different ways.
"""

import Yomchi.Environment as Env
import Yomchi.preprocessing as data
import Yomchi.visualization as ui

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf


def data_exploration():
    """
    @details
    <h1> Data Exploration:</h1>
    <h2> Setup </h2>
    <ul>
        <li> Initialize Environment.
        <li> Initialize Tensorflow session.
        <li> Set seed for Numpy and Tensorflow
        <li> Specify run configuration parameters.
        <li> Specify dataset generation parameters.
    </ul>
    <h2> Procedure </h2>
    <ul>
    <li> <b>Step 1:</b>
        <ul>
        <li> Import LFP data.
        <li> Plot channels 0 and 97.
        <li> Plot the first 10 seconds of LFP data from selected channel at original sampling rate (1250Hz).
        </ul>
    <li> <b>Step 2:</b>
        <ul>
        <li> Import angles data.
        <li> Plot angles data.
        <li> Plot 20 seconds at minute 6 of angles data at original sampling rate (39.06Hz).
        </ul>
    <li> <b>Step 3:</b>
        <ul>
        <li> If the chosen synchronization method is to downsample the LFP data:
            <ul>
            <li> Downsample LFP data to match Angles sampling rate.
            <li> Plot channels 0 and 97 downsampled.
            <li> Plot the first 10 seconds of LFP data from selected channel at position's sampling rate (39.06Hz).
            </ul>
        <li> Else, the chosen synchronization method is to upsample the Angles data:
            <ul>
            <li> Fill angles gaps to match the LFP sampling rate.
            <li> Plot angles data in the new sampling rate.
            </ul>
        </ul>
    <li> <b>Step 4:</b>
        <ul>
        <li> Interpolate angles data using a 'interpolation' approach.
        <li> Plot angles data after interpolation.
        <li> Plot 20 seconds at minute 6 of angles data at LFP's sampling rate (39.06Hz). Interpolated with the
        specified method.
        </ul>
    <li> <b>Step 5:</b>
        <ul>
        <li> Label data by concatenating LFPs and interpolated Angles in a single 2D-array.
        <li> Plot LFP and angles data.
        <li> Print an angles data window where the first LED is unsynchronized and then the second LED is lost too.
        </ul>
    <li> <b>Step 6:</b>
        <ul>
        <li> Clean the labeled dataset from invalid values.
        <li> Print Bar plot for labeled angles.
        </ul>
    <li> <b>Step 7:</b>
        <ul>
        <li> Count number of NaNs at the beginning and at the end of the data.
        <li> Print some NaN counts in different dataset stages.
        </ul>
    <li> <b>Step 8:</b> Plot clean LFP data from channels 0 and interpolated angles data.
    <li> <b>Step 9:</b> Plot Boxplot of all channels in two figures, from 0-49 and from 50-98.
    <li> <b>Step 10:</b>
        <ul>
        <li> Plot Boxplot of interpolated angles.
        <li> Plot Boxplot of interpolated angles vs channel 0
        </ul>
    <li> <b>Finish Test and Exit:</b> Save logs, captures and results.
    </ul>
    """

    # ------- SETUP -------
    Env.init_environment(True)

    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    # Run configuration parameters
    PLOT = True

    # Session and methods Parameters
    session = 765  # 771 or 765

    interpolation = "quadratic"  # "linear" "quadratic" "cubic" "nearest" "Shortest Angle"
    sync_method = "Downsample LFPs"  # "Upsample Angles" "Downsample LFPs"

    # Data Properties
    num_channels = data.EC014_41_NUMBER_OF_CHANNELS
    rate_used = data.POSITION_DATA_SAMPLING_RATE
    if sync_method == "Upsample Angles":
        rate_used = data.LFP_DATAMAX_SAMPLING_RATE

    # Windowing properties
    lfp_channel = 70
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
                             "LFP Signal channel to use": lfp_channel}

    parameters_dictionary.update(extra)
    Env.print_parameters(parameters_dictionary)

    # ------- STEP 1 -------
    Env.step(f"Importing LFP data from session: {session}")
    lfp_data = data.load_lfp_data(data.LFP[session])

    if PLOT:
        Env.print_text(f"Plot LFP data from channels 0 and 97 at {data.LFP_DATAMAX_SAMPLING_RATE}Hz")
        figname = f"{session}_LFP_C0_and_C97_{data.LFP_DATAMAX_SAMPLING_RATE}Hz"
        plt.figure(figname)
        plt.subplot(211)
        plt.plot(lfp_data[:, 0], "xr")
        plt.title(f"Señal LFP. Sesión: {session} del Canal 0 a {data.LFP_DATAMAX_SAMPLING_RATE}Hz")

        plt.subplot(212)
        plt.plot(lfp_data[:, 97], "xb")
        plt.title(f"Señal LFP. Sesión: {session} del Canal 97 a {data.LFP_DATAMAX_SAMPLING_RATE}Hz")
        ui.store_figure(figname)

        Env.print_text(f"Plot the first 10 seconds of "
                       f"LFP data from channel {lfp_channel} at {data.LFP_DATAMAX_SAMPLING_RATE}Hz")
        figname = f"{session}_LFP-window_C{lfp_channel}_{data.LFP_DATAMAX_SAMPLING_RATE}Hz"
        plt.figure(figname)
        plt.plot(lfp_data[0: round(10 * data.LFP_DATAMAX_SAMPLING_RATE), lfp_channel], ".b", markersize=3)
        plt.title(f"Primeros 10s de Señal LFP, sesión: {session}, canal: {lfp_channel} a "
                  f"{data.LFP_DATAMAX_SAMPLING_RATE}Hz")
        ui.store_figure(figname)

    # ------- STEP 2 -------
    Env.step(f"Importing angles data from session: {session}")

    angles_data = data.load_angles_data(data.ANGLES[session])

    Env.print_text("Min angle: " + str(np.nanmin(angles_data)) + ", Max angle: " + str(np.nanmax(angles_data)))

    if PLOT:
        Env.print_text(f"Plot Angles data [°] at {data.POSITION_DATA_SAMPLING_RATE}Hz")
        figname = f"{session}_Angles_degrees_{data.POSITION_DATA_SAMPLING_RATE}Hz"
        plt.figure(figname)
        plt.plot(angles_data[:], "xr")
        plt.title(f"Información de ángulos [°]. Sesión: {session} a {data.POSITION_DATA_SAMPLING_RATE}Hz")
        ui.store_figure(figname)

        Env.print_text(f"Plot 20 seconds at minute 6 of Angles data at {data.POSITION_DATA_SAMPLING_RATE}Hz")
        figname = f"{session}_Angles_degrees-window_{data.POSITION_DATA_SAMPLING_RATE}Hz"
        plt.figure(figname)
        plt.plot(angles_data[round(6 * 60 * data.POSITION_DATA_SAMPLING_RATE):
                             round((6 * 60 + 20) * data.POSITION_DATA_SAMPLING_RATE)], ".r", markersize=4)
        plt.title(f"20s al minuto 6 de los ángulos, sesión: {session} a {data.POSITION_DATA_SAMPLING_RATE}Hz")
        ui.store_figure(figname)

    # ------- STEP 3 -------
    if sync_method == "Downsample LFPs":
        Env.step("Downsample LFP data to match Angles sampling rate.")

        lfp_data = data.downsample_lfps(lfp_data, data.LFP_DATAMAX_SAMPLING_RATE, data.POSITION_DATA_SAMPLING_RATE)

        if PLOT:
            Env.print_text(f"Plot LFP downsampled data to {rate_used} Hz from channels 0 and 97.")
            figname = f"{session}_LFP_downsampled_0_and_97_{rate_used}Hz"
            plt.figure(figname)
            plt.subplot(211)
            plt.plot(lfp_data[:, 0], "xr")
            plt.title(f"Señal LFP del Canal 0 a {rate_used}Hz. Sesión: {session}")

            plt.subplot(212)
            plt.plot(lfp_data[:, 97], "xb")
            plt.title(f"Señal LFP del Canal 97 a {rate_used}Hz. Sesión: {session}")
            ui.store_figure(figname)

            Env.print_text(f"Plot the first 10 seconds of LFP downsampled data from channel {lfp_channel} "
                           f"at {rate_used}Hz")
            figname = f"{session}_LFP-window-down_C{lfp_channel}_{rate_used}Hz"
            plt.figure(figname)
            plt.plot(lfp_data[0: round(10 * rate_used), lfp_channel], ".b", markersize=3)
            plt.title(f"Primeros 10s de Señal LFP, sesión: {session}, canal: {lfp_channel} a "
                      f"{rate_used}Hz")
            ui.store_figure(figname)

    elif sync_method == "Upsample Angles":
        Env.step("Upsample Angles data to reach a higher sampling rate")

        angles_data = data.angles_expansion(angles_data, data.POSITION_DATA_SAMPLING_RATE,
                                            data.LFP_DATAMAX_SAMPLING_RATE)

        if PLOT:
            Env.print_text("Plot Angles data expanded with 'NaN' [°]")
            figname = f"{session}Expanded_Angles_degrees_{rate_used}Hz"
            plt.figure(figname)
            plt.plot(angles_data[:], "xb")
            plt.title(f"Información de ángulos [°] expandida a {rate_used}Hz. Sesión: {session}")
            ui.store_figure(figname)

    # ------- STEP 4 -------
    Env.step(f"Interpolate angles data using a {interpolation} approach.", 4)
    angles_data_interpolated = data.interpolate_angles(angles_data, interpolation)

    interpolation_ = ""
    if sync_method == "Upsample Angles":
        interpolation_ = f".\n Interpolada con: {interpolation}"

    Env.print_text(f"Number of NaNs in the Angles Data before interpolation: {np.count_nonzero(np.isnan(angles_data))}")
    Env.print_text(f"Number of -1s in the Angles Data before interpolation: {np.count_nonzero(angles_data == -1)}")

    Env.print_text(f"Number of NaNs in the Angles Data after interpolation: "
                   f"{np.count_nonzero(np.isnan(angles_data_interpolated))}")
    Env.print_text(f"Number of -1s in the Angles Data after interpolation: "
                   f"{np.count_nonzero(angles_data_interpolated == -1)}")

    if PLOT and sync_method == sync_method == "Upsample Angles":
        Env.print_text(f"Plot Angles data after {interpolation} interpolation [°]")
        figname = f"{session}_Angles_{interpolation} _degrees_{rate_used}Hz"
        plt.figure(figname)
        plt.plot(angles_data_interpolated[:], "xr")
        plt.title(f"Información de ángulos [°]. Sesión: {session}{interpolation_} a {rate_used}Hz")
        ui.store_figure(figname)

        Env.print_text(f"Plot 20 seconds at minute 6 of Angles data interpolated with {interpolation} at {rate_used}Hz")
        figname = f"{session}_Angles_degrees-window_up_{rate_used}Hz"
        plt.figure(figname)
        plt.plot(angles_data_interpolated[round(6*60*rate_used): round((6*60+20) * rate_used)], ".r", markersize=4)
        plt.title(f"20s al minuto 6 de los ángulos, sesión: {session} a {rate_used}Hz."
                  f"\n Interpolada con {interpolation}")
        ui.store_figure(figname)

    # ------- STEP 5 -------
    Env.step("Label data by concatenating LFPs and interpolated Angles in a single 2D-array.", 5)

    # IF round_angles: Rounding the angles to be discrete labels, starting from 'base_angle' until 360 on steps of
    # 'offset_between_angles'. Else Not rounding the angles to be discrete labels
    labeled_data = data.add_labels(lfp_data, np.expand_dims(angles_data_interpolated, axis=1), round_angles,
                                   base_angle, offset_between_angles)

    if PLOT:
        Env.print_text(f"Plot LFP data from channels 0 and interpolated angles data at {rate_used}Hz. [°]")
        figname = f"{session}_LFP_C0_and_angles_{interpolation}_{rate_used}Hz"
        plt.figure(figname)
        plt.subplot(211)
        plt.plot(labeled_data[:-1, 0], "xr")
        plt.title(f"Señal LFP del Canal 0. Muestreada a {rate_used}Hz. Sesión: {session}")

        plt.subplot(212)
        plt.plot(labeled_data[:, -1], "xb")
        plt.title(f"Información de ángulos [°]. Sesión: {session}{interpolation_} a {rate_used}Hz")
        ui.store_figure(figname)

    # ------- STEP 6 -------
    Env.step("Clean the labeled dataset from invalid values.")

    clean_dataset = data.clean_invalid_positional(labeled_data, sync_method == "Upsample Angles")

    if round_angles:
        labels = np.arange(base_angle, 360, offset_between_angles)
        percentages = []
        for u in range(base_angle, 360, offset_between_angles):
            percentages.append(round(np.sum(clean_dataset[:, -1] == u) * 100 / len(clean_dataset[:, -1])))

        labels_percent = np.concatenate((np.expand_dims(labels, axis=1), np.expand_dims(percentages, axis=1)), axis=1)
        dataframe_labels = pd.DataFrame(data=labels_percent, columns=["Angulos", "Porcentaje"])

        if PLOT:
            Env.print_text(f"Plot Barplot of Labels after {interpolation} interpolation")
            figname = f"{session}_BarPLotAngles_{interpolation}_{rate_used}Hz"
            plt.figure(figname)
            sns.barplot(x="Angulos", y="Porcentaje", data=dataframe_labels)
            plt.title(f"Gráfico de Barras de las etiquetas. Sesión: {session}{interpolation_} a "
                      f"{rate_used}Hz")
            ui.store_figure(figname)

    # ------- STEP 7 -------
    Env.step("Count and print number of NaNs in the Dataset.")

    invalid_begin = 0
    invalid_end = 0
    length = len(angles_data_interpolated)
    for i in range(length):
        if angles_data_interpolated[i] != -1:
            invalid_begin = i
            break
    for i in range(length - 1, 0, -1):
        if angles_data_interpolated[i] != -1:
            invalid_end = i
            break
    invalid_end = length - (invalid_end + 1)

    Env.print_text(
        f"Number of invalid in Angles Data without interpolation: {np.count_nonzero(angles_data == -1)}")
    Env.print_text("Number of invalid in Labeled Dataset with interpolated Angles Data: "
                   + str(np.count_nonzero(angles_data_interpolated == -1)))
    Env.print_text("Number of invalid in Labeled and Clean Dataset with interpolated Angles Data: "
                   + str(np.count_nonzero(clean_dataset[0][:, -1] == -1)))

    if PLOT:
        # ------- STEP 8 -------
        Env.step(f"Plot clean LFP data from channels 0 and interpolated angles data at {rate_used}Hz. [°]")

        figname = f"{session}_LFP_C{lfp_channel}_clean_and_angles_{interpolation}_{rate_used}Hz"
        plt.figure(figname)
        plt.subplot(211)
        plt.plot(clean_dataset[0][:, lfp_channel], "xr")
        plt.title(f"Señal LFP del Canal {lfp_channel}. Muestreada a {rate_used}Hz. Sesión: {session}")
        plt.subplot(212)
        plt.plot(clean_dataset[0][:, -1], "xb")
        plt.title(f"Información de ángulos [°]. Sesión: {session}{interpolation_} a "
                  f"{rate_used}Hz.")
        ui.store_figure(figname)

    if PLOT:
        # ------- STEP 9 -------
        Env.step("Plot Boxplot of all channels in two figures, from 0-49 and from 50-98.")

        figname = f"{session}_BoxPLot0-49_{interpolation}_{rate_used}Hz"
        plt.figure(figname, figsize=[12, 16])
        sns.boxplot(data=clean_dataset[0][:, 0:50], orient="h")
        plt.ylabel("Canales")
        plt.xlabel("Voltaje")
        plt.title(f"Diagrama de caja de los canales 0-49.\nSesión: {session}{interpolation_}"
                  f" a {rate_used}Hz")
        ui.store_figure(figname)

        figname = f"{session}_BoxPLot50-98_{interpolation}_{rate_used}Hz"
        plt.figure(figname, figsize=[12, 16])
        sns.boxplot(data=clean_dataset[0][:, 50:99], orient="h")
        plt.ylabel("Canales")
        plt.xlabel("Voltaje")
        plt.title(f"Diagrama de caja de los canales 50-98.\nSesión: {session}{interpolation_}"
                  f" a {rate_used}Hz")
        ui.store_figure(figname)

        # ------- STEP 10 -------
        Env.step("Plot Boxplot of interpolated angles.")

        figname = f"{session}_BoxPLotAngles_{interpolation}_{rate_used}Hz"
        plt.figure(figname)
        sns.boxplot(x=clean_dataset[0][:, -1])
        plt.ylabel("Muestras")
        plt.xlabel("Ángulos")
        plt.title(f"Diagrama de caja de los ángulos.\nSesión: {session}{interpolation_} "
                  f"a {rate_used}Hz")
        ui.store_figure(figname)

    # ------- FINISH TEST AND EXIT -------
    #Env.finish_test()
    Env.finish_test(f"{session}_{interpolation}_" + sync_method.replace(" ", ""))


# ------- Execute Dataset Exploration -------
data_exploration()
