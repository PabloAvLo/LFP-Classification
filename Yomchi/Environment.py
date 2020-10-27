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
import numpy as np

# Module Constants
START_TIME = datetime.datetime.now()
CURRENT_FOLDER = os.getcwd()
RESULTS_FOLDER = CURRENT_FOLDER + "/Results-" + START_TIME.strftime("%Y-%m-%d_%H-%M") + "/"
LOGS_FOLDER = RESULTS_FOLDER + "Logs/"
CAPTURES_FOLDER = RESULTS_FOLDER + "Captures/"
VERSIONS_LOG_PATH = LOGS_FOLDER + "versions_log.txt"
YOMCHI_LOG_PATH = LOGS_FOLDER + "yomchi_log.txt"
LINE_LENGTH = 100

# Module Variables
step_number = 1
versions_log = None
yomchi_log = None
caller_file = None
debug = False


def init_environment(print_the_header=False, enable_debug=False):
    """
    Initialize the directories and files to log the results.
    @param print_the_header: If True, print the header
    @param enable_debug: If True, run some code to help debugging
    """
    global versions_log
    global yomchi_log
    global caller_file
    global debug

    debug = enable_debug
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

    if print_the_header:
        print_header()

    log_versions()


def print_text(text):
    """
    Print a passed text and save it in the yomchi_log.txt.
    @param text: Text to print
    @return None
    """
    global yomchi_log

    print(text)
    yomchi_log.write(text + "\n")


def print_box(text):
    """
    Print a passed text in a box and save it in the yomchi_log.txt.
    @param text: Text to print
    @return None
    """
    text_length = len(text)
    counter = 0
    for i in range(0, text_length):
        if text[i] == '\n':
            counter = 0
        else:
            counter += 1

        if counter == LINE_LENGTH:
            if text[i] in ['a', 'e', 'i', 'o', 'u']:
                text = text[:i - 1] + '-' + '\n' + text[i - 1:]
            else:
                text = text[:i] + '\n' + text[i:]
            counter = 0

    text_array = text.splitlines()
    print_text("\n|" + "-" * (LINE_LENGTH + 2) + "|")
    for line in text_array:
        print_text("| " + line.ljust(LINE_LENGTH, ' ') + " |")
    print_text("|" + "-" * (LINE_LENGTH + 2) + "|")


def step(description, new_step_number=0):
    """
    Print the current step number with a brief description.
    @param description: Description of the step.
    @param new_step_number: If defined, reset the steps numeration to the passed value.
    @return None
    """
    global step_number

    larger_string = 7
    pad = round((LINE_LENGTH - larger_string) / 2)

    if new_step_number != 0:
        step_number = new_step_number

    print_box(" " * pad + "Step " + str(step_number) + "\n" + description)
    step_number += 1


def print_parameters(params_dictionary):
    """
    Print a list of parameters.
    @param params_dictionary: Dictionary of parameters {name: value} to print.
    @return None
    """

    pad = round((LINE_LENGTH - 18) / 2)
    title = " " * pad + "*** Parameters ***"

    names = list(params_dictionary.keys())
    values = list(params_dictionary.values())
    longest_name = max(names, key=len)

    new_names = []
    for name in names:
        new_names.append(name + " " * (len(longest_name) - len(name)) + " : ")

    new_dictionary = np.char.add(new_names, values)

    print_text("\n|" + "-" * (LINE_LENGTH + 2) + "|")
    print_text("| " + title.ljust(LINE_LENGTH, ' ') + " |")
    print_text("| " + ''.ljust(LINE_LENGTH, ' ') + " |")
    for param in new_dictionary:
        print_text("| " + param.ljust(LINE_LENGTH, ' ') + " |")
    print_text("|" + "-" * (LINE_LENGTH + 2) + "|\n")


def log_versions():
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


def print_header():
    """
    Print a header in the console output and log files.
    @return None
    """
    larger_string = 33
    pad = round((LINE_LENGTH - larger_string) / 2)
    header = "#" * LINE_LENGTH + "#" * 3 \
             + "\n#" + " " * pad + "                                 " + " " * pad + "#" \
             + "\n#" + " " * pad + "    University of Costa Rica     " + " " * pad + "#" \
             + "\n#" + " " * pad + "Electrical Engineering Department" + " " * pad + "#" \
             + "\n#" + " " * pad + "          Grade Thesis           " + " " * pad + "#" \
             + "\n#" + " " * pad + "      Pablo Avila [B30724]       " + " " * pad + "#" \
             + "\n#" + " " * pad + "    jose.avilalopez@ucr.ac.cr    " + " " * pad + "#" \
             + "\n#" + " " * pad + "                                 " + " " * pad + "#" \
             + "\n" + "#" * LINE_LENGTH + "#" * 3 \
             + "\n" + " " * pad + "     *** START OF TEST ***  "\
             + "\n# File      : " + caller_file \
             + "\n# Date      : " + START_TIME.strftime("%m-%d-%Y") \
             + "\n# Start Time: " + START_TIME.strftime("%H:%M:%S") + "\n"
    print_text(header)
    versions_log.write(header)

def finish_test(rename_results_folder=None):
    """
    Safely finish the test run and log some final information.
    @param rename_results_folder: Optional name for the results folder.
    @return None
    """
    global yomchi_log
    global versions_log

    larger_string = 33
    pad = round((LINE_LENGTH - larger_string) / 2)

    finish_time = datetime.datetime.now()
    elapsed_time = finish_time - START_TIME

    days = elapsed_time.days
    hours, rem = divmod(elapsed_time.seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    footer = "\n" + "#" * LINE_LENGTH + "#" * 3 \
             + "\n" + " " * pad + "     *** END OF TEST ***  " \
             + "\n# Test Duration : " + str(days) + str(hours) + ":" + str(minutes) + ":" + str(seconds) \
             + "\n# Finish Time   : " + finish_time.strftime("%H:%M:%S") \
             + "\n" + "#" * LINE_LENGTH + "#" * 3

    print_text(footer)

    yomchi_log.close()
    versions_log.close()

    if rename_results_folder is not None:
        os.rename(RESULTS_FOLDER, CURRENT_FOLDER + "/" + rename_results_folder + "/")

    sys.exit()
