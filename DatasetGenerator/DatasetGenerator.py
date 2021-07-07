# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file DatasetGenerator.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date July, 2021
@details Properly loads the input data and labels and prepare a clean dataset to finally export it as a pickle file.
"""

from Yomchi import \
    Environment as Env, \
    preprocessing as data, \
    visualization as ui

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle


def data_set_generator():
    """
    @details
    <h1> Dataset Generator:</h1>
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
        <li> <b>Step 1:</b> Import LFP data.
        <li> <b>Step 2:</b> Import angles data.
        <li> <b>Step 3:</b>
        <ul>
            <li> Downsample LFP data to match the Angles sampling rate or
            <li> Fill angles gaps to match the LFP sampling rate.
        </ul>
        <li> <b>Step 4:</b> Label data by concatenating LFPs and Angles in a single 2D-array.
        <li> <b>Step 5:</b> Clean the labeled dataframe from -1 values, which represent the wrongly acquired
        positional samples.
        <li> <b>Step 6:</b> Interpolate angles data using an 'interpolation' approach.
        <li> <b>Step 7:</b> Plotting Angles and the selected channel of the LFPs after cleaning and interpolation.
        <li> <b>Step 8:</b>
        <ul>
            <li> Get preferred angle according to LFPs channel.
            <li> Plotting the sum of LFPs per angle from 0 to 359 degrees.
        </ul>
        <li> <b>Step 9:</b> Get largest subset of data.
        <li> <b>Step 10:</b> Split the subset into training, validation and testing set.
        <li> <b>Step 11:</b> Save the dataset to a pickle file.
        <li> <b>Step 12:</b> Convert data to windowed series.
        <li> <b>Finish Test and Exit: </b> Save logs, captures, results and the generated dataset into a .pickle file.
    </ul>
    """

    # ------- SETUP -------
    Env.init_environment(True, True)

    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    # Dataset Generation Parameters
    # The recording session 771 has a 23.81% of invalid positions, while the session 765 has only 6.98%
    session = 771  # 771 or 765
    interpolation = "Shortest Angle"  # "linear" "quadratic" "cubic" "nearest" "Shortest Angle"
    sync_method = "Downsample LFPs"  # "Upsample Angles" "Downsample LFPs"

    # Data Properties
    num_channels = data.EC014_41_NUMBER_OF_CHANNELS
    rate_used = data.POSITION_DATA_SAMPLING_RATE
    if sync_method == "Upsample Angles":
        rate_used = data.LFP_DATAMAX_SAMPLING_RATE

    # Windowing properties
    window_size = 4  # Recommended between 100ms to 200ms.
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

    o_pickle_file_name = f"S-{session}_C{lfp_channel}_I-{interpolation}_F-{rate_used}_W-" \
                         f"{int(window_size * 1e3 / rate_used)}ms.pickle"

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

    # ------- STEP 1 -------
    Env.step(f"Importing LFP data from session: {session}")
    lfp_data = data.load_lfp_data(data.LFP[session])

    # ------- STEP 2 -------
    Env.step(f"Importing angles data from session: {session}")
    angles_data = data.load_angles_data(data.ANGLES[session])

    # ------- STEP 3 -------
    interpolation_ = ""
    if sync_method == "Downsample LFPs":
        Env.step("Downsample LFP data to match Angles sampling rate.")
        lfp_data = data.downsample_lfps(lfp_data, data.LFP_DATAMAX_SAMPLING_RATE, data.POSITION_DATA_SAMPLING_RATE)

    elif sync_method == "Upsample Angles":
        Env.step("Upsample Angles data to reach a higher sampling rate")
        angles_data = data.angles_expansion(angles_data, data.POSITION_DATA_SAMPLING_RATE,
                                            data.LFP_DATAMAX_SAMPLING_RATE)
        interpolation_ = f".\n Interpolada con: {interpolation}"

    # ------- STEP 4 -------
    Env.step("Label data by concatenating LFPs and interpolated Angles in a single 2D-array.")
    # IF round_angles: Rounding the angles to be discrete labels, starting from 'base_angle' until 360 on steps of
    # 'offset_between_angles'. Else, the angles are not rounded to discrete labels.
    labeled_data = data.add_labels(lfp_data, np.expand_dims(angles_data, axis=1), round_angles,
                                   base_angle, offset_between_angles)

    # ------- STEP 5 -------
    Env.step("Clean the labeled dataset from discontinuities in positional data (angles).")

    clean_datasets = data.clean_invalid_positional(labeled_data)

    # ------- STEP 6 -------
    Env.step(f"Interpolate angles data using a {interpolation} approach.")

    # Get all LFP channels, excluding the angles.
    clean_interpolated_data = [s[:, :-1] for s in clean_datasets]

    # Interpolate angles and add them to the dataset
    clean_interpolated_angles = [data.interpolate_angles(s[:, -1], interpolation) for s in clean_datasets]

    for i in range(len(clean_interpolated_data)):
        clean_interpolated_data[i] = \
            np.concatenate((clean_interpolated_data[i], np.expand_dims(clean_interpolated_angles[i], axis=1)), axis=1)

    # ------- STEP 7 -------
    Env.step("Plotting Angles and one channel of the LFPs after cleaning and interpolation.")

    Env.print_text(f"Plotting Angles data [°] at {rate_used}Hz")
    figname = f"{session}_Angles_degrees_{int(rate_used)}Hz"
    plt.figure(figname)
    plt.plot(clean_interpolated_data[4][:, -1], "xr")
    plt.title(f"Información de ángulos [°]. Sesión: {session} a {rate_used}Hz")
    ui.store_figure(figname, show=Env.debug)

    Env.print_text(f"Plotting LFPs Channel {lfp_channel} at {rate_used}Hz")

    figname = f"{session}_LFP_c{lfp_channel}_{int(rate_used)}Hz"
    plt.figure(figname)
    plt.plot(clean_interpolated_data[4][:, lfp_channel], "x-r")
    plt.title(f"LFPs del Canal {lfp_channel}. Sesión: {session} a {rate_used}Hz")
    ui.store_figure(figname, show=Env.debug)

    Env.step(f"Plot clean LFP data from the chosen channel and interpolated angles data at {rate_used}Hz. [°]")

    figname = f"{session}_LFP_C{lfp_channel}_clean_and_angles_{interpolation}_{rate_used}Hz"
    plt.figure(figname)
    plt.subplot(211)
    plt.plot(clean_interpolated_data[0][0:round(5*60*rate_used), lfp_channel], "xr")
    plt.title(f"Señal LFP del Canal {lfp_channel}.")
    plt.subplot(212)
    plt.plot(clean_interpolated_data[0][0:round(5*60*rate_used), -1], "xb")
    plt.title(f"Información de ángulos [°]. {interpolation_}.")
    plt.suptitle(f"5 minutos limpios de la sesión: {session} a {rate_used}Hz.")
    ui.store_figure(figname)

    # ------- STEP 8 -------
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
    figname = f"{session}_preferred_angle_LFP_c{lfp_channel}_{int(rate_used)}Hz"
    plt.figure(figname)
    plt.plot(range(360), angles, "x-r")
    plt.title(f"Suma de LFPs del Canal {lfp_channel} por ángulo. Sesión: {session}.")
    plt.xlabel(f"Ángulo en grados")
    ui.store_figure(figname, show=Env.debug)

    # ------- STEP 9 -------
    Env.step("Get largest subset of data.")

    max_length = 0
    max_subset_index = 0
    for index, subset in enumerate(clean_interpolated_data):
        if max_length < len(subset):
            max_length = len(subset)
            max_subset_index = index

    largest_subset = clean_interpolated_data[max_subset_index]
    Env.print_text(f"Shape of the largest subset of data : [{len(largest_subset)},{len(largest_subset[0])}]")

    # ------- STEP 10 -------
    Env.step()

    n = len(largest_subset)
    train_array = largest_subset[0:int(n * 0.7), :]
    valid_array = largest_subset[int(n * 0.7):int(n * 0.9), :]
    test_array = largest_subset[int(n * 0.9):, :]

    Env.print_text(f"Training data shape: [{len(train_array)},{len(train_array[0])}]")
    Env.print_text(f"Validation data shape: [{len(valid_array)},{len(valid_array[0])}]")
    Env.print_text(f"Test data shape: [{len(test_array)},{len(test_array[0])}]")

    # ------- STEP 11 -------
    Env.step(f"Save the dataset to pickle file: {o_pickle_file_name}.")

    with open(f"{Env.RESULTS_FOLDER}/{o_pickle_file_name}", 'wb') as f:
        pickle.dump([train_array, valid_array, test_array], f)

    # ------- STEP 12 -------
    Env.step("Convert data to windowed series.")

    train_data = data.channels_to_windows(train_array, lfp_channel, window_size, batch_size, shuffle_buffer)
    val_data = data.channels_to_windows(valid_array, lfp_channel, window_size, batch_size, shuffle_buffer)
    test_data = data.channels_to_windows(test_array, lfp_channel, window_size, batch_size, shuffle_buffer)

    # The shape should be (batch, time, features) to be compatible with what tensorflow expects as default.
    for example_inputs, example_labels in train_data.take(1):
        Env.print_text(f'Inputs shape (batch, time, samples): {example_inputs.shape}')
        Env.print_text(f'Labels shape (batch, time, labels): {example_labels.shape}')

    # ------- FINISH TEST AND EXIT -------
    Env.finish_test()
    #Env.finish_test(str(session) + "_" + interpolation.title() + "_" + sync_method.replace(" ", ""))


# ------- Execute Dataset generation -------
data_set_generator()
