# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file visualization.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date July, 2021
@details This module contains a set of functions to plot and in general terms, visualize data.
"""

import matplotlib.pyplot as plt
import Yomchi.Environment as Env


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
