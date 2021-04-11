# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file visualization.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details This module contains a set of functions to plot and in general terms, visualize data.
"""

import matplotlib.pyplot as plt
import Yomchi.Environment as Env
import numpy as np

# Matplotlib figures parameters
FIG_FORMAT = 'png'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['savefig.format'] = FIG_FORMAT
plt.rcParams['lines.markersize'] = 1
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlepad'] = 7
plt.rcParams['axes.grid'] = False


def store_figure(fig_name, show=False):
    """
    Stores a figure.
    @param fig_name: Name of the figure to store.
    @param show: Displays the figure when ready. Warning: Stalls execution until closing it.
    @return None
    """

    plt.figure(fig_name)
    plt.savefig(fname=Env.CAPTURES_FOLDER + fig_name + "." + FIG_FORMAT)
    if show:
        plt.show()
    plt.close(fig_name)


def plot_predictions(actual_data, predictions, max_subplots=3):
    angles = actual_data[:, 1]
    plt.figure(figsize=(12, 8))
    max_n = min(max_subplots, len(angles))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel('Angles')

        plt.scatter(angles[n, :, 0], edgecolors='k', label='angles', c='#2ca02c', s=64)
        plt.scatter(predictions[n, :, 0], mmarker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()
