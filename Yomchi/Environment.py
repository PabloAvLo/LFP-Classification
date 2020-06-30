# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file Environment.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date May, 2020
@details Python environment which integrates data pre-processing, ML models implementation, data visualization and
metrics. This module contains a set of functions to successfully document every test run by creating Results folders
with console logs, generated images, dependencies versions, etc.
"""

import os
import sys
import inspect
import platform
import datetime

# Module Constants
START_TIME = datetime.datetime.now()
CURRENT_FOLDER = os.getcwd()
RESULTS_FOLDER = CURRENT_FOLDER + "/Results-" + START_TIME.strftime("%Y-%m-%d_%H-%M") + "/"
LOGS_FOLDER = RESULTS_FOLDER + "Logs/"
CAPTURES_FOLDER = RESULTS_FOLDER + "Captures/"
VERSIONS_LOG_PATH = LOGS_FOLDER + "versions_log.txt"
YOMCHI_LOG_PATH = LOGS_FOLDER + "yomchi_log.txt"

# Module Variables
step_number = 1
versions_log = None
yomchi_log = None
caller_file = None


def initEnvironment(print_header=False):
    """
    Initialize the directories and files to log the results.
    @param print_header: If True, print the header
    """
    global versions_log
    global yomchi_log
    global caller_file

    os.makedirs(RESULTS_FOLDER)
    os.makedirs(LOGS_FOLDER)
    os.makedirs(CAPTURES_FOLDER)

    # First get the full filename (including path and file extension)
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename

    # Now get rid of the directory and extension
    caller_file = os.path.basename(caller_filename)

    yomchi_log = open(YOMCHI_LOG_PATH, "w+")

    versions_log = open(VERSIONS_LOG_PATH, "w+")

    if print_header:
        printHeader()

    logVersions()


def printText(text):
    """
    Print a passed text and save it in the yomchi_log.txt.
    @param text: Text to print
    @return None
    """
    global yomchi_log

    print(text)
    yomchi_log.write(text + "\n")


def printBox(text):
    """
    Print a passed text in a box and save it in the yomchi_log.txt.
    @param text: Text to print
    @return None
    """
    line_length = 64
    text_length = len(text)
    counter = 0
    for i in range(0, text_length):
        if text[i] == '\n':
            counter = 0
        else:
            counter += 1

        if counter == 64:
            if text[i] in ['a', 'e', 'i', 'o', 'u']:
                text = text[:i - 1] + '-' + '\n' + text[i - 1:]
            else:
                text = text[:i] + '\n' + text[i:]
            counter = 0

    text_array = text.splitlines()
    printText("|" + "-" * (line_length + 2) + "|")
    for line in text_array:
        printText("| " + line.ljust(line_length, ' ') + " |")
    printText("|" + "-" * (line_length + 2) + "|")


def step(description, newstep_number=0):
    """
    Print the current step number with a brief description.
    @param description: Description of the step.
    @param newstep_number: If defined, reset the steps numeration to the passed value.
    @return None
    """
    global step_number

    if newstep_number != 0:
        step_number = newstep_number

    printBox(" " * 28 + "Step " + str(step_number) + "\n" + description)
    step_number += 1


def logVersions():
    """
    Stores the environment versions including packages, libraries and OS.
    @return None
    """
    global versions_log
    required_packages = ["Keras", "numpy", "pandas", "matplotlib", "scikit-learn", "seaborn",
                         "sklearn", "tensorflow", "pip"]

    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    try:
        distro = platform.linux_distribution()
    except:
        distro = ["N/A", ""]

    versions_log.write("\n\nOperation System: ")
    versions_log.write("\n  Kernel: " + platform.system() + " " + platform.release())
    versions_log.write("\n  Distribution: " + distro[0] + " " + distro[1])

    versions_log.write("\n\nPython version: " + str(sys.version_info[0]) + "."
                       + str(sys.version_info[1]) + "." + str(sys.version_info[2]))
    versions_log.write("\n\nPackages versions:\n")

    list = freeze.freeze()

    versions_log.write("  Required Packages:")
    for package in list:
        for required in required_packages:
            if package.find(required) != -1:
                index = package.find("==")
                package = package.replace("==", " " * (25 - index) + "=  ")
                versions_log.write("\n    " + package)

    list = freeze.freeze()
    versions_log.write("\n\n  Complete List:")
    for package in list:
        index = package.find("==")
        package = package.replace("==", " " * (25 - index) + "=  ")
        versions_log.write("\n    " + package)


def printHeader():
    line_length = 63
    header = "#" * line_length \
             + "\n#" + " " * 14 + "                                 " + " " * 14 + "#" \
             + "\n#" + " " * 14 + "    University of Costa Rica     " + " " * 14 + "#" \
             + "\n#" + " " * 14 + "Electrical Engineering Department" + " " * 14 + "#" \
             + "\n#" + " " * 14 + "          Grade Thesis           " + " " * 14 + "#" \
             + "\n#" + " " * 14 + "      Pablo Avila [B30724]       " + " " * 14 + "#" \
             + "\n#" + " " * 14 + "    jose.avilalopez@ucr.ac.cr    " + " " * 14 + "#" \
             + "\n#" + " " * 14 + "                                 " + " " * 14 + "#" \
             + "\n" + "#" * line_length \
             + "\n" + " " * 14 + "     *** START OF TEST ***  "\
             + "\n# File: " + caller_file \
             + "\n# Date: " + START_TIME.strftime("%m-%d-%Y") \
             + "\n# Time: " + START_TIME.strftime("%H:%M:%S") + "\n"
    printText(header)
    versions_log.write(header)
