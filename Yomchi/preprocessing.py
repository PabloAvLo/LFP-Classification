# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file preprocessing.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details This module contains a set of functions to import clean, parse and reshape input data.
"""

import os
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
import Yomchi.Environment as Env

# Dataset Paths
PATH_TO_DATASETS = os.path.join(Env.CURRENT_FOLDER, "../Datasets/")
LFP_765     = PATH_TO_DATASETS + "ec014.765.eeg"
ANGLES_765  = PATH_TO_DATASETS + "ec014.765.whl"
LFP_771     = PATH_TO_DATASETS + "ec014.771.eeg"
ANGLES_771  = PATH_TO_DATASETS + "ec014.771.whl"

# Sampling Rates
RAW_DATAMAX_SAMPLING_RATE   = 20000  ## 20 kHz
RAW_NEURALYNX_SAMPLING_RATE = 32552  ## 32.552 kHz
LFP_DATAMAX_SAMPLING_RATE   = 1250   ## 1.25 kHz
LFP_NEURALYNX_SAMPLING_RATE = 1252   ## 1.252 kHz
POSITION_DATA_SAMPLING_RATE = 39.06  ## 39.06 Hz
EC014_41_NUMBER_OF_CHANNELS = 99     ## 16 shank probes with 8 electrodes each, minus bad channels.


def load_lfp_data(file=LFP_771, channels=EC014_41_NUMBER_OF_CHANNELS):
    """
    Loads the LFP signals from a .eeg file (LFP Data only f < 625Hz) or a .dat file (LFP Data + Spikes).
    @param file: Path to the file containing the animal LFP data.
    @param channels: Number of recorded channels in LFP signals file.
    @return lfp: Array (n x channels) with the data. With the columns being the channels and the rows the
    a different time step.
    """
    Env.print_text("Loading LFP Data of " + str(channels) + " channels from: " + os.path.basename(file) + ".")

    signals = open(file, "rb")
    signalsArray = np.fromfile(file=signals, dtype=np.int16)
    lfp = np.reshape(signalsArray, (-1, channels))
    signals.close()

    return lfp


def load_angles_data(file=ANGLES_771, degrees=True):
    """
    Loads the animal position data from a .whl file which contain 2 (x, y) pairs, one for each LED. If any position
    value equals '-1' then it's replaced with 'NaN' instead.
    @param file: Path to the file containing the animal LED's position information
    @param degrees: If this flag is set, then the angles are returned in degrees from [0, 360[, or radians otherwise.
    @return angles: Array with the angles in radians extracted from the positions. The angles are given as
    float16 values calculated as arctan(y2 - y1 /x2 - x1). Unless the denominator is 0, in that case '0' is returned
    for that element.
    """
    Env.print_text("Loading the animal position data from: " + os.path.basename(file) + ".")

    positions_file = open(file, "rb")
    positions = np.genfromtxt(fname=positions_file, dtype=np.float16, delimiter='\t')
    positions_file.close()

    positions[positions == -1] = np.NaN
    angles = np.arctan2((np.subtract(positions[:, 3], positions[:, 1])),
                        (np.subtract(positions[:, 2], positions[:, 0])))

    if degrees:
        angles = np.degrees(angles)
        angles += 180

    return angles


def angles_expansion(angles_data, orig_rate, new_rate):
    """
    Fill angular data with 'NaN' values to match an expected sampling rate. Usually the position data is acquired at a
    lower sampling rate than the LFP signals.
    @details Assuming that the acquisition of data started and stopped at the same time, then no data has to be added at
    after the last sample.
    @param angles_data: Array with the angles data extracted from the animal positions.
    @param orig_rate: Sampling rate originally used to acquire the data.
    @param new_rate: New sampling rate of the data. The gaps are filled with 'NaN'.
    @return upsampled_data: Original data filled with zeros to match the new sampling rate.
    """
    Env.print_text("Expanding angles data with 'NaN' values to match the new sampling rate: " + str(new_rate) + "Hz. "
                   + "Original was: " + str(orig_rate) + "Hz.")

    expansion_factor = round(new_rate/orig_rate)
    padding = np.full((len(angles_data), expansion_factor - 1), np.NaN)
    upsampled_data = np.concatenate((np.transpose(np.array([angles_data])), padding), axis=1)
    upsampled_data = upsampled_data[:-1, :]
    upsampled_data = upsampled_data.flatten()

    return upsampled_data


def slerp(start, end, amount):
    """
    Spherical Linear Interpolation: This is equivalent to the linear interpolation, applied in a sphere.
    It considers the 'start' and 'end' as angles in a circumference where the objective is to find the smallest arch.
    param start: Start angle with values from 0 to 359
    param end: Final angle with values from 0 to 359
    param amount: Value between [0, 1] which determines how close the interpolated angle will be placed from the Start
    angle (0) or from the Final angle (1), being 0.5 the middle.
    return interpolated_angle: Interpolated angle between 'start' and 'end'.
    """
    shortest_angle = ((end-start) + 180) % 360 - 180
    interpolated_angle = (start + shortest_angle * amount) % 360
    return interpolated_angle


def vectorized_slerp(angles_data):
    """
    Replace 'NaN' values between valid values (interpolation) in angular data with an interpolated value using
    Spherical Linear Interpolation (SLERP).
    param angles_data: Array with the angles data extracted from the animal positions.
    return interpolated_angles: Array with the angles data interpolated using SLERP
    """
    start_angle = np.nan
    no_nans = 0
    first_nan_index = 0
    interpolated_angles = angles_data

    for i in range(1, len(interpolated_angles)):
        # If a valid value followed by NaN: this is the first NaN, start counting
        if np.isnan(interpolated_angles[i]) and not np.isnan(interpolated_angles[i-1]):
            start_angle = interpolated_angles[i-1]
            no_nans = 1
            first_nan_index = i

        # If a NaN followed by another NaN: Increment counter 1+.
        elif np.isnan(interpolated_angles[i]) and np.isnan(interpolated_angles[i - 1]):
            no_nans += 1

        # If a NaN followed by a valid value: This is the last NaN, interpolate.
        elif not np.isnan(interpolated_angles[i]) and np.isnan(interpolated_angles[i - 1]):
            if no_nans > 0 and not np.isnan(start_angle):
                amount = 0
                end_angle = interpolated_angles[i]
                for j in range(first_nan_index, first_nan_index + no_nans):
                    amount += 1/(no_nans + 1)
                    interpolated_angles[j] = slerp(start_angle, end_angle, amount)
                start_angle = np.nan

    return interpolated_angles


def interpolate_angles(angles_data, method="linear"):
    """
    Replace 'NaN' values in angular data with an interpolated value using a given method.
    @param angles_data: Array with the angles data extracted from the animal positions.
    @param method: Interpolation method to fill the gaps in the data. The optional methods available are the supported
    by pandas.DataFrame.interpolate function, which are: 'linear', 'quadratic', 'cubic', 'polynomial', among others.
    @return interpolated_angles: Array with the angles data interpolated using the given method.
    """
    Env.print_text("Interpolate angles data using " + method + " method.")

    if method == "SLERP":
        interpolated_angles = vectorized_slerp(angles_data)

    else:
        angles_series = pd.Series(angles_data)
        interpolated_angles = angles_series.interpolate(method)
        interpolated_angles = interpolated_angles.to_numpy()

    return interpolated_angles


def downsample_lfps(lfp_data, orig_rate, new_rate):
    """
    Drop samples to match the new sampling rate. Usually the the LFP signals are acquired at a higher sampling rate
    than the position data.
    @param lfp_data: Matrix [n x numChannels] with the LFP signals
    @param orig_rate: Sampling rate originally used to acquire the data.
    @param new_rate: New sampling rate of the data. The 'extra' samples are not included in the returned data.
    @return resampled_data: Original data downsampled to the new rate. It only returns the indices that are multiples of
    the rates reason.
    """
    Env.print_text("Downsampling LFP data by dropping 'extra' samples to match the new sampling rate: " + str(new_rate)
                   + "Hz. Original was: " + str(orig_rate) + "Hz.")

    rate_reason = round(orig_rate/new_rate)
    resampled_data = lfp_data[::rate_reason]
    resampled_data = np.append(resampled_data, [lfp_data[-1]], axis=0)

    return resampled_data


def add_labels(lfps, angles, round_labels, start=0, offset=30):
    """
    Add an additional column to the LFP signals matrix with the angular data used as the labels.
    @param lfps: Matrix [n x numChannels] with the LFP signals used as the preliminary features of the data.
    @param angles: Array with the angles data extracted from the positions used as the labels of the data.
    @param round_labels: Boolean, if true the labels are rounded to angles multiples of 'offset' starting from 'start'
    @param start: Angle in [0°, 360°[ used as first label.
    @param offset: Offset in [1°, 360°[ between labels starting from 'start' angle.
    @return labeled_data: Matrix with the labeled data [n x lfps[numChannels], angles].
    """
    Env.print_text("Adding labels to the data by concatenating the [" + str(len(lfps)) + " x " + str(len(lfps[0])) +
                   "] LFP data matrix with the [" + str(len(angles)) + "] Angles vector.")
    Env.print_text("Rounding Labels = " + str(round_labels))

    if round_labels:
        if (0 <= start < 360) and (1 <= offset < 360):
            labels = np.arange(start, 360, offset)

        for i in range(0, len(angles)):
            if not np.isnan(angles[i]):
                minval = np.inf
                label = start
                for tag in labels:
                    diff = abs(angles[i] - tag)
                    if diff < minval:
                        minval = diff
                        label = tag
                angles[i] = label

    labeled_data = np.concatenate((lfps, angles), axis=1)

    return labeled_data


def clean_unsync_boundaries(labeled_dataset, is_dataframe=True):
    """
    Clean the data rows which have 'NaN' values as labels (angles) from the beginning and end of the data.
    @details At the beginning and at the end of the recording sessions, the LEDs of position are unsynchronized,
    hence '-1' values are used instead to denote invalid position data. These values were replaced by 'NaN' and are
    meant to be removed from the data since they are not representative labels.
    @param labeled_dataset: Matrix [n x (numChannels +1)] with the LFP signals used as the preliminary features of the
    data and the angles data extracted from the positions used as the labels of the data.
    @param is_dataframe: If set, manage the input labeled dataset as a Pandas Dataframe, or a Numpy array otherwise.
    @return clean_dataset: Input data without 'NaN' values in the beginning nor the end.
    """

    if is_dataframe:
        clean_dataset = labeled_dataset[~np.isnan(labeled_dataset["Angle"])]
    else:
        clean_dataset = labeled_dataset[~np.isnan(labeled_dataset[:, -1])]

    return clean_dataset


def ndarray_to_dataframe(dataset, rate):
    """
    Converts an n-D Numpy array to a Pandas Dataframe
    @param dataset: Matrix [n x (numChannels +1)] with the LFP signals used as the preliminary features
    of the data and the angles data extracted from the positions used as the labels of the data.
    @param rate:
    @return dataframe: Pandas data frame with Electrode_0-99 and Angles as columns names, and the timestamp as indeces
    calculated as 1/rate * 1e6 to get the time step of the acquisition in microseconds.
    """

    columns = []
    for i in range(0, 99):
        columns.append("E" + str(i))
    columns.append("Angle")

    time_step = round((1/rate) * 1e6)   # 1/f [us]: 1250Hz => 800us, 39.006Hz => 25602us
    indeces = np.arange(0, len(dataset) * time_step, time_step)
    dataframe = pd.DataFrame(data=dataset, columns=columns, index=indeces)

    return dataframe


def channels_to_windows(series, channel, window_size, batch_size, shuffle_buffer=None):
    """
    Receives a numpy array containing the time series of LFP signals of n channels and returns the same data, separated
    in windows.
    @param series: Numpy Array with the LFP data of the n channels.
    @param channel: Channel to use.
    @param window_size: Size of the windows in which the data are being split.
    @param batch_size: Number of pairs data-labels to group as a batch
    @param shuffle_buffer: Number of windows to shuffle at the same time.
    @return windowed_ds: LFP data of the selected channel separated in windows.
    """

    data = series[:, channel]
    labels = series[window_size::, -1]

    # Creates a dataset from the input
    windowed_data = tf.data.Dataset.from_tensor_slices(data)

    # Split the data set in windows shifting each window by 1 and forcing them to the same size (window_size + 1)
    windowed_data = windowed_data.window(window_size, shift=1, drop_remainder=True)

    # Make each window a numpy array row.
    windowed_data = windowed_data.flat_map(lambda window: window.batch(window_size))

    if Env.debug:
        print("Channel " + str(channel) + " Data:")
        for window in windowed_data.take(5):
            print(window.numpy())

    # Get the average angle for each window.
    labels = average_angles(labels, window_size)

    # Add labels to the data
    windowed_ds = tf.data.Dataset.zip((windowed_data, labels))

    if shuffle_buffer != None:
        # Shuffle the data in groups of shuffle_buffer to accelerate. Instead of shuffle it all at once.
        windowed_ds = windowed_ds.shuffle(shuffle_buffer)

    # Batch the data into sets of 'batch_size'.
    windowed_ds = windowed_ds.batch(batch_size).prefetch(1)

    if Env.debug:
        for x, y in windowed_ds.take(5):
            print("x = ", x.numpy())
            print("y = ", y.numpy())

    return windowed_ds


def average_angles(angles, window_size):
    """
    Receives a numpy array containing the time series of the angles and returns the an set of windows with the average
    of the 'window_size' angles in each window. Each window is 1 element shifted from the previous window.
    @param angles: Numpy Array with the Angles data to use as the labels.
    @param window_size: Size of the windows in which the data are being split.
    @return average_angles: Averaged Angles data separated in windows.
    """

    # Get an array of labels from the series
    windowed_angles = tf.data.Dataset.from_tensor_slices(angles)

    # Split the data set in windows shifting each window by 1 and forcing them to the same size (window_size + 1)
    windowed_angles = windowed_angles.window(window_size, shift=1, drop_remainder=True)

    # Make each window a numpy array row.
    windowed_angles = windowed_angles.flat_map(lambda window: window.batch(window_size))
    averaged_angles = windowed_angles.map(lambda window: tf.math.reduce_mean(window))

    if Env.debug:
        print("Angles:")
        for window in windowed_angles.take(5):
            print(window.numpy())
        for window in averaged_angles.take(5):
            print(window.numpy())

    return averaged_angles

