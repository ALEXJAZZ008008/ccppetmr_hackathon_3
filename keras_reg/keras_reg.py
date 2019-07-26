from __future__ import division, print_function
import os
import re
from tensorflow import keras as k
import numpy as np
import sirf.STIR as PET


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    return [atoi(c) for c in re.split(r'(\d+)', string)]


# https://stackoverflow.com/questions/36000843/scale-numpy-array-to-certain-range
def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def get_x_train(input_path, input_prefix):
    x_train = []
    x_train_fixed = []
    x_train_moving_fixed = []

    relative_path = input_path + "/fixed/"
    x_train_fixed_files = os.listdir(relative_path)
    x_train_fixed_files.sort(key=human_sorting)

    for i in range(len(x_train_fixed_files)):
        if len(x_train_fixed_files[i].split(input_prefix)) > 1:
            x_train_fixed.append(
                rescale_linear(
                    PET.ImageData(relative_path + x_train_fixed_files[i]).as_array().squeeze(), 0, 1))

    relative_path = input_path + "/moving/"
    x_train_moving_files = os.listdir(relative_path)
    x_train_moving_files.sort(key=human_sorting)

    for i in range(len(x_train_moving_files)):
        temp_relative_path = relative_path + x_train_moving_files[i] + '/'
        x_train_moving_files_fixed_files = os.listdir(temp_relative_path)
        x_train_moving_files_fixed_files.sort(key=human_sorting)
        x_train_moving = []

        for j in range(len(x_train_moving_files_fixed_files)):
            if len(x_train_moving_files_fixed_files[j].split(input_prefix)) > 1:
                x_train_moving.append(
                    rescale_linear(
                            PET.ImageData(
                                temp_relative_path + x_train_moving_files_fixed_files[j]).as_array().squeeze(), 0, 1))

        x_train_moving_fixed.append(x_train_moving)

    for i in range(len(x_train_moving_fixed)):
        for j in range(len(x_train_moving_fixed[i])):
            x_train.append(np.asarray([x_train_fixed[i], x_train_moving_fixed[i][j]]).T)

    return np.nan_to_num(np.asarray(x_train)).astype(np.float)


def get_y_train(input_path):
    y_train = []

    with open(input_path + "/transforms.csv", 'r') as file:
        for line in file:
            line = line.rstrip()
            line_tuple = line.split(',')
            line_float = []

            for i in range(len(line_tuple)):
                line_float.append(float(line_tuple[i]))

            y_train.append(line_float)

    return np.nan_to_num(np.asarray(y_train))


def fit_model(test_bool, load_bool, apply_bool, input_path, input_prefix, output_path):
    if test_bool:
        # random data for now
        x_train = np.random.rand(100, 100, 100, 2)  # 100 images, shape (100, 100), channels static & moving
        y_train = np.random.rand(100, 3) * 2 - 1  # 100 3-vectors
    else:
        x_train = get_x_train(input_path, input_prefix)
        y_train = get_y_train(input_path)

    # normalisation: change dtype (speed)
    x_train = x_train.astype(np.float)
    y_train = y_train.astype(np.float)
    x_train /= x_train.std(axis=(2, 3))[:, :, None, None]  # ensures unit stdev

    if load_bool:
        model = k.models.load_model(output_path + "/model.h5")
    else:
        input_x = k.layers.Input(x_train.shape[1:])

        # 5 x 5 x (previous num channels = 2) kernels (32 times => 32 output channels)
        # padding='same' for zero padding
        x = k.layers.Conv2D(32, 3, activation=k.activations.relu, padding='same')(input_x)
        # 3 x 3 x (previous num channels = 32) kernels (64 times)
        x = k.layers.Conv2D(64, 3, activation=k.activations.relu, padding='same')(x)
        x = k.layers.Dropout(0.20)(x)  # discard 20% outputs
        x = k.layers.MaxPooling2D(2)(x)  # downsample by factor of 2, using "maximum" interpolation
        x = k.layers.Flatten()(x)  # vectorise
        x = k.layers.Dense(128, activation=k.activations.relu)(x)  # traditional neural layer with 128 outputs
        x = k.layers.Dropout(0.20)(x)  # discard 20% outputs
        x = k.layers.Dense(64, activation=k.activations.relu)(x)  # traditional neural layer with 64 outputs
        x = k.layers.Dropout(0.20)(x)  # discard 20% outputs
        x = k.layers.Dense(32, activation=k.activations.relu)(x)  # traditional neural layer with 32 outputs
        x = k.layers.Dropout(0.20)(x)  # discard 20% outputs
        x = k.layers.Dense(96, activation=k.activations.relu)(x)  # traditional neural layer with 96 outputs
        x = k.layers.Dropout(0.20)(x)  # discard 20% outputs
        x = k.layers.Dense(3, activation=k.activations.tanh)(x)  # 3 outputs

        model = k.Model(input_x, x)

        # N x 3

        # losses:  K.losses.*
        # optimisers: K.optimizers.*
        model.compile(optimizer=k.optimizers.Adam(), loss=k.losses.mse)

    model.summary()
    model.fit(x_train, y_train, epochs=250, verbose=1)

    loss = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', loss)

    model.save(output_path + "/model.h5")

    if apply_bool:
        output = model.predict(x_train)
        difference = y_train - output

        print("Max difference: " + str(difference.max()))
        print("Mean difference: " + str(difference.mean()))

        with open(output_path + "/output_transforms.csv", 'w') as file:
            for i in range(len(output)):
                file.write(str(output[i][0]) + ',' + str(output[i][1]) + ',' + str(output[i][0]) + '\n')

        with open(output_path + "/difference.csv", 'w') as file:
            for i in range(len(difference)):
                file.write(str(difference[i][0]) + ',' + str(difference[i][1]) + ',' + str(difference[i][0]) + '\n')


def test_model(test_bool, data_input_path, data_input_prefix, model_input_path, output_path):
    if test_bool:
        # random data for now
        x_test = np.random.rand(100, 100, 100, 2)  # 100 images, shape (100, 100), channels static & moving
    else:
        x_test = get_x_train(data_input_path, data_input_prefix)

    model = k.models.load_model(model_input_path + "/model.h5")
    output = model.predict(x_test)

    with open(output_path + "/output_transforms.csv", 'w') as file:
        for i in range(len(output)):
            file.write(str(output[i][0]) + ',' + str(output[i][1]) + ',' + str(output[i][0]) + '\n')


fit_model(False, False, True, "../training_data/", ".nii", "../results/")
