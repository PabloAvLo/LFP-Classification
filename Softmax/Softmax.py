# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file Softmax.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details Feed-forward Neural Network using Softmax algorithm in the output layer.
"""

import Yomchi.Environment as Env
import Yomchi.preprocessing as data
import Yomchi.visualization as ui

import matplotlib.pyplot as plt
import numpy as np


"""
<h1> Feed-Forward Neural Network </h1>
<ol>
<li> Step 1
<ul>
<li> Initialize Environment
<li> Define parameters... 
</ul>
"""

Env.initEnvironment(True)

"""
<li> Step 2
<ul>
<li> Import LFP and angles data
<li> Plot a channels 0 and 97.
</ul>
</ol>
"""

Env.step("Importing LFP and angles data")
W = data.loadLFPData()

Env.printText("Plotting LFP data from channels 0 and 97")
plt.figure("LFP_Channels_0_and_97", figsize=ui.FIG_SIZE, dpi=ui.DPI)
plt.subplot(211)
plt.plot(W[:, 0], "r")
plt.title("Señal LFP del Canal 0")

plt.subplot(212)
plt.plot(W[:, 97], "b")
plt.title("Señal LFP del Canal 97")
ui.storeFigure("LFP_Channels_0_and_97", "LFP_C0-C97.png", True)
