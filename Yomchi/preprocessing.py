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
import Yomchi.Environment as Env

# Dataset Paths
PATH_TO_DATASETS = os.path.join(Env.CURRENT_FOLDER, "../Datasets/")
LFP_765    = PATH_TO_DATASETS + "ec014.765.eeg"
ANGLES_765 = PATH_TO_DATASETS + "ec014.765.whl"
LFP_771    = PATH_TO_DATASETS + "ec014.771.eeg"
ANGLES_771 = PATH_TO_DATASETS + "ec014.765.whl"

# Sampling Rates
RAW_DATAMAX_SAMPLING_RATE   = 20000  ## 20 kHz
RAW_NEURALYNX_SAMPLING_RATE = 32552  ## 32.552 kHz
LFP_DATAMAX_SAMPLING_RATE   = 1250   ## 1.25 kHz
LFP_NEURALYNX_SAMPLING_RATE = 1252   ## 1.252 kHz
POSITION_DATA_SAMPLING_RATE = 39.06  ## 39.06 Hz
EC014_41_NUNBER_OF_CHANNELS = 99     ## 16 shank probes with 8 electrodes each, minus bad channels.

# Data Properties
numChannels         = 99
lfp_SamplingRate    = LFP_DATAMAX_SAMPLING_RATE
angles_SamplingRate = POSITION_DATA_SAMPLING_RATE

def loadLFPData(file =LFP_771, numChannels = 99, rate = LFP_DATAMAX_SAMPLING_RATE):
    """
    Loads the LFP signals from a .eeg file (LFP Data only f < 625Hz) or .dat files (LFP Data + Spikes).
    @param file: Path to the file containing the animal LFP data.
    @param numChannels: Number of recorded channels in LFP signals file.
    @param rate: Sampling rate of the recorded LFP signals.
    @return data: Numpy Array (n x numChannels) with the data. With the columns being the channels and the rows the
    time step.
    """
    Env.printText("Loading LFP Data of " + str(numChannels) + " channels sampled at " + str(rate) + "Hz from: "
                  + os.path.basename(file) + ".")

    # Read it as signed shorts (16 bits signed)
    # Reshape it to (n x numChannels)
    signals = open(LFP_765, "rb")
    signalsArray = np.fromfile(file=signals, dtype=np.int16)
    data = np.reshape(signalsArray, (-1, numChannels))
    length = len(data)

    return data


def loadAnglesData(file =ANGLES_771, rate = POSITION_DATA_SAMPLING_RATE):
    """
    Loads the animal position data from a .whl file which contain 2 (x, y) pairs, one for each LED.
    @param file: Path to the file containing the animal LED's position information
    @param rate: Sampling rate of the acquired positions.
    @return data: Numpy Array with the angles data extracted from the positions.
    """
    Env.printText("Loading LFP Data of " + numChannels + "channels sampled at " + rate + "Hz from: " + file + ".")
    #angles = open(ANGLES_765, "rb")
    #anglesArray = np.genfromtxt(fname=angles, dtype=np.float16, delimiter='\t')

    #print(anglesArray[100, 0])
