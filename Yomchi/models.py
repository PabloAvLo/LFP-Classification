# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file models.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details This module contains a set of functions to generate Machine Learning models such as Feed-forward Neural
Networks, Recurrent Neural Networks (RNN), Convolutional Neural Networks (CNN) and more.
"""

import numpy as np
import sklearn.metrics
import keras.models
import keras.layers
import keras.utils
import keras.regularizers