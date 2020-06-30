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

# Figures parameters

DPI = 200
FIG_SIZE = [12, 9]
FIG_FORMAT = "png"


def storeFigure(figName, fileName, show=False):
    plt.figure(figName)
    plt.savefig(fname=Env.CAPTURES_FOLDER+fileName, dpi=DPI, format=FIG_FORMAT)
    if show:
        plt.show()
    plt.close(figName)
