# #######################################################################
#                   University of Costa Rica
#               Electrical Engineering Department
#                         Grade Thesis
# #######################################################################

"""
@file Predictor.py
@author Pablo Avila [B30724] jose.avilalopez@ucr.ac.cr
@copyright MIT License
@date July, 2021
@details The predictor loads a dataset of LFP with Angles as labels previously generated and feed it to a Neural
Network in order to train it to predict new angles based on LFP data windows.
"""

from Yomchi import \
    Environment as Env, \
    preprocessing as data, \
    models, \
    visualization as ui

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle


def predictor():
    """
    @details
    <h1> Predictor </h1>
    <h2> Experiment Setup </h2>
    <ul>
        <li> Initialize Environment
        <li> Initialize Tensorflow session.
        <li> Set seed for Numpy and Tensorflow
        <li> Specify run configuration parameters.
        <li> Specify run configuration parameters.
        <li> Specify dataset generation parameters.
        <li> Specify model's training parameters.
    </ul>
    <h2> Procedure </h2>
    <ul>
        <li> <b>Step 1:</b> Import training, validation and test datasets of LFPs as inputs and Angles as labels.
        <li> <b>Step 2:</b> Convert data to windowed series.
        <li> <b>Step 3:</b> Create the model with the specified parameters
        <li> <b>Step 4:</b> Train the model with the training and validation data.
        <li> <b>Step 5:</b> Plotting the original angular data vs the predictions for different batch sizes.
        <li> <b>Step 6:</b> Plotting the original angular data vs the predictions of all testing data.
        <li> <b>Step 7:</b> Plotting the historic of the training vs the validation error values.
        <li> <b>Finish Test and Exit:</b> Save logs, captures and results.
    </ul>
    """

    # ------- SETUP -------
    Env.init_environment(True)

    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    tf.random.set_seed(51)
    np.random.seed(51)

    # Session, data and methods Parameters
    session = 771  # 765 or 771
    interpolation = "Shortest Angle"  # or "linear", "quadratic", "cubic", "nearest", "Shortest Angle"
    rate_used = data.POSITION_DATA_SAMPLING_RATE  # data.LFP_DATAMAX_SAMPLING_RATE or data.POSITION_DATA_SAMPLING_RATE

    # Windowing properties
    window_size = int(rate_used * 0.1 * 11)  # Frequency_used * 0.1 = 100ms.
    batch_size = 32
    shuffle_buffer = 1000
    lfp_channel = 40
    batches_to_plot = [5, 10, 20, 50]

    # Parameters
    units_per_layer = 32
    dropout = 0.2
    layers = 3
    epochs = 20
    model_name = "LSTM"  # "LSTM" "MLP" or "CNN"

    # Input pickle file.
    i_pickle_file_name = f"S-{session}_I-" + interpolation.replace(" ", "") + f"_F-{int(rate_used)}.pickle"
    parameters_dictionary = {"Recording Session Number": str(session),
                             "Interpolation Method": interpolation.title(),
                             "Sampling Rate to Use": str(rate_used) + "Hz",
                             "LFP Signal channel to use": lfp_channel,
                             "Shuffle buffer size": str(shuffle_buffer),
                             "Pickle file used": i_pickle_file_name,
                             "Batches to Plot": str(batches_to_plot),
                             "Batch size (# of windows)": batch_size,
                             "Window Size": f"{window_size} samples ({int(window_size * 1e3 / rate_used)}ms)",
                             "Model Name": model_name,
                             "Layers (for MLP model only)": str(layers),
                             "Epochs": str(epochs),
                             "Units per Layer": str(units_per_layer),
                             "Dropout Regularization %": str(dropout)
                             }

    Env.print_parameters(parameters_dictionary)

    # ------- STEP 1 -------
    Env.step("Import training, validation and test datasets of LFPs as inputs and Angles as labels.")

    with open(f"{data.PATH_TO_DATASETS}Pickles/{i_pickle_file_name}", "rb") as f:
        train_array, valid_array, test_array = pickle.load(f)

    # ------- STEP 2 -------
    Env.step("Convert data to windowed series.")

    # The train data is shuffled so the network is trained with a variety of examples.
    train_data = data.channels_to_windows(train_array, lfp_channel, window_size, batch_size, shuffle_buffer)

    # The validation and testing data are not shuffled because the intention is to predict the series in order.
    val_data = data.channels_to_windows(valid_array, lfp_channel, window_size, batch_size)
    test_data = data.channels_to_windows(test_array, lfp_channel, window_size, batch_size)

    for example_inputs, example_labels in train_data.take(1):
        Env.print_text(f'Inputs shape (batch, time, channels): {example_inputs.shape}')
        Env.print_text(f'Labels shape (batch, time, labels): {example_labels.shape}')

    # ------- STEP 3 -------
    Env.step("Create the model with the specified parameters ")

    if model_name == "LSTM":
        model = models.lstm(units_per_layer, dropout)
    elif model_name == "CNN":
        model = models.cnn(window_size, units_per_layer, dropout)
    else:
        model = models.mlp(layers, units_per_layer, dropout)

    for example_inputs, example_labels in train_data.take(1):
        Env.print_text(f'Input shape: {example_inputs.shape}')
        Env.print_text(f'Output shape: {model(example_inputs).shape}')

    # ------- STEP 4 -------
    Env.step("Train the model with the training and validation data.")

    history = models.compile_and_fit(model, train_data, val_data, epochs=epochs, patience=epochs)

    Env.print_text(f"\nTraining metrics history:")
    Env.print_text(f"Loss: {history.history['loss']}")
    Env.print_text(f"Mean Absolute Error: {history.history['mean_absolute_error']}")
    Env.print_text(f"\nValidation metrics history:")
    Env.print_text(f"Loss: {history.history['val_loss']}")
    Env.print_text(f"Mean Absolute Error: {history.history['val_mean_absolute_error']}")

    # ------- STEP 5 -------
    Env.step(f"Plotting the original angular data vs the predictions of {str(batches_to_plot)} batches")

    lfps = []
    real = []
    pred = []

    interpolated = ""
    if rate_used == data.LFP_DATAMAX_SAMPLING_RATE:
        interpolated = f"Interpolación: {interpolation}"
        if interpolation == "Shortest Angle":
            interpolated = "Interpolación: Por ángulo corto."

    for batches in batches_to_plot:
        for inputs, label in test_data.take(batches):
            # Getting original data
            real = np.append(real, label[:, 0, 0].numpy())
            lfps = np.append(lfps, inputs[:, 0, 0].numpy())
            # Getting predicted data
            pred = np.append(pred, model.predict(inputs))

        # Plotting
        figname = f"{model_name}_S-{session}_W-{int(window_size * 1e3 / rate_used)}_F-{int(rate_used)}_E-{epochs}_C-" \
                  f"{lfp_channel}_B{batches}"

        fig, axs = plt.subplots(2, 1, num=figname)
        axs[0].set_title(
            f"{model_name} fs:{rate_used} Epochs:{epochs} Batches:{batches} Ventana:"
            f"{int(window_size * 1e3 / rate_used)}ms."
            f" {interpolated}")

        axs[0].set_xlabel(f"Predicción de Ángulos contra ángulos originales.")
        axs[0].scatter(range(len(real)), real, label='Originales', c='#2ca02c', s=10)
        axs[0].scatter(range(len(pred)), pred, marker='X', label='Predicciones', c='#ff7f0e', s=16)
        axs[0].legend()

        axs[1].set_xlabel(f'Canal de LFP: {lfp_channel}')
        axs[1].scatter(range(len(lfps)), lfps, label='LFPs', c='#0a7fdb', s=10)
        axs[1].legend()
        ui.store_figure(figname, show=Env.debug)

    for inputs, label in test_data:
        # Getting original data
        real = np.append(real, label[:, 0, 0].numpy())
        lfps = np.append(lfps, inputs[:, 0, 0].numpy())
        # Getting predicted data
        pred = np.append(pred, model.predict(inputs))

    # ------- STEP 6 -------
    Env.step(f"Plotting the original angular data vs the predictions of all testing data")

    figname = f"{model_name}_S-{session}_W-{int(window_size * 1e3 / rate_used)}_F-{int(rate_used)}_E-{epochs}_C-" \
              f"{lfp_channel}_BAll"

    fig, axs = plt.subplots(2, 1, num=figname)
    axs[0].set_title(
        f"{model_name} fs:{rate_used} Epochs:{epochs} Batches: Todos. Ventana:"
        f"{int(window_size * 1e3 / rate_used)}ms."
        f" {interpolated}")

    axs[0].set_xlabel(f"Predicción de Ángulos contra ángulos originales.")
    axs[0].scatter(range(len(real)), real, label='Originales', c='#2ca02c', s=10)
    axs[0].scatter(range(len(pred)), pred, marker='X', label='Predicciones', c='#ff7f0e', s=16)
    axs[0].legend()

    axs[1].set_xlabel(f'Canal de LFP: {lfp_channel}')
    axs[1].scatter(range(len(lfps)), lfps, label='LFPs', c='#0a7fdb', s=10)
    axs[1].legend()
    ui.store_figure(figname, show=Env.debug)

    # ------- STEP 7 -------
    Env.step(f"Plotting the historic of the training vs the validation error values.")

    figname = f"{model_name}_S-{session}_W-{int(window_size * 1e3 / rate_used)}_F-{int(rate_used)}_E-{epochs}_C-" \
              f"{lfp_channel}_history"
    plt.figure(figname)
    plt.plot(history.history['mean_absolute_error'], "x-r", label="Entrenamiento")
    plt.plot(history.history['val_mean_absolute_error'], "x-b", label="Validación")
    plt.xlabel("Época")
    plt.ylabel("MAE")
    plt.legend(loc='best')
    plt.title(f"Error Medio contra épocas. {model_name} fs:{rate_used} Epochs:{epochs} Ventana:"
              f"{int(window_size * 1e3 / rate_used)}ms. {interpolated}")
    ui.store_figure(figname)

    # ------- FINISH TEST AND EXIT -------
    # Env.finish_test()
    Env.finish_test(f"M-{model_name}_S-{session}_W-{int(window_size * 1e3 / rate_used)}_F-{int(rate_used)}"
                    f"_E-{epochs}_C-{lfp_channel}")


# ------- Execute Predictor -------
predictor()
