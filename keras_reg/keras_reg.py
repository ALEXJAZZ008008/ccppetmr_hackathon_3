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
    return [atoi(c) for c in re.split(r"(\d+)", string)]


# https://stackoverflow.com/questions/36000843/scale-numpy-array-to-certain-range
def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def get_x(input_path, input_prefix):
    print("Getting x")

    x = []
    x_fixed = []
    x_moving_fixed = []

    relative_path = input_path + "/fixed/"
    x_fixed_files = os.listdir(relative_path)
    x_fixed_files.sort(key=human_sorting)

    print("Get x fixed")

    for i in range(len(x_fixed_files)):
        if len(x_fixed_files[i].split(input_prefix)) > 1:
            x_fixed.append(
                rescale_linear(
                    PET.ImageData(relative_path + x_fixed_files[i]).as_array().squeeze(), 0, 1))

    print("Got x fixed")

    relative_path = input_path + "/moving/"
    x_moving_files = os.listdir(relative_path)
    x_moving_files.sort(key=human_sorting)

    print("Get x moving")

    for i in range(len(x_moving_files)):
        temp_relative_path = relative_path + x_moving_files[i] + "/"
        x_moving_files_fixed_files = os.listdir(temp_relative_path)
        x_moving_files_fixed_files.sort(key=human_sorting)
        x_moving = []

        for j in range(len(x_moving_files_fixed_files)):
            if len(x_moving_files_fixed_files[j].split(input_prefix)) > 1:
                x_moving.append(
                    rescale_linear(
                            PET.ImageData(
                                temp_relative_path + x_moving_files_fixed_files[j]).as_array().squeeze(), 0, 1))

        x_moving_fixed.append(x_moving)

    print("Got x moving")

    for i in range(len(x_moving_fixed)):
        for j in range(len(x_moving_fixed[i])):
            x.append(np.asarray([x_fixed[i], x_moving_fixed[i][j]]).T)

    print("Got x")

    return np.nan_to_num(np.asarray(x)).astype(np.float)


def get_y(input_path):
    print("Get y")

    y = []

    with open(input_path + "/transforms.csv", "r") as file:
        for line in file:
            line = line.rstrip()
            line_tuple = line.split(",")
            line_float = []

            for i in range(len(line_tuple)):
                line_float.append(float(line_tuple[i]))

            y.append(line_float)

    print("Got y")

    return np.nan_to_num(np.asarray(y))


def perceptron(x):
    x = k.layers.Flatten()(x)

    return x


def fully_connected(x):
    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=256)(x)
    x = k.layers.Activation("relu")(x)

    return x


def deep_fully_connected(x):
    x = k.layers.Flatten()(x)

    for _ in range(2):
        x = k.layers.Dense(units=256)(x)
        x = k.layers.Activation("relu")(x)

    return x


def papernet(x):
    for _ in range(4):
        x = k.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = k.layers.Activation("relu")(x)

        x = k.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(x)

    for _ in range(3):
        x = k.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
        x = k.layers.Activation("relu")(x)

    x = k.layers.Flatten()(x)

    return x


def alexnet_module(x):
    # Convolutional Layer
    x = k.layers.Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding="valid")(x)
    x = k.layers.Activation("relu")(x)
    # Max Pooling
    x = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

    # Convolutional Layer
    x = k.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(1, 1), padding="valid")(x)
    x = k.layers.Activation("relu")(x)
    # Max Pooling
    x = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

    # Convolutional Layer
    x = k.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = k.layers.Activation("relu")(x)

    # Convolutional Layer
    x = k.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = k.layers.Activation("relu")(x)

    # Convolutional Layer
    x = k.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = k.layers.Activation("relu")(x)
    # Max Pooling
    x = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

    return x


def alexnet(x):
    # Up Sampling
    x = k.layers.UpSampling2D(size=(3, 3))(x)

    # Split
    x_1 = alexnet_module(x)
    x_2 = alexnet_module(x)

    # Add
    x = k.layers.Add()([x_1, x_2])

    # Flatten
    x = k.layers.Flatten()(x)

    # Fully Connected Layer
    x = k.layers.Dense(units=4096)(x)
    x = k.layers.Activation("relu")(x)

    # Fully Connected Layer
    x = k.layers.Dense(units=4096)(x)
    x = k.layers.Activation("relu")(x)

    # Fully Connected Layer
    x = k.layers.Dense(units=1000)(x)
    x = k.layers.Activation("relu")(x)

    return x


def googlenet_module(x, conv_filter_1, conv_filter_2, conv_filter_3, conv_filter_4, conv_filter_5, conv_filter_6):
    x_1 = k.layers.Conv2D(filters=conv_filter_1, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x_1 = k.layers.Activation("relu")(x_1)

    x_2 = k.layers.Conv2D(filters=conv_filter_2, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x_2 = k.layers.Activation("relu")(x_2)
    x_2 = k.layers.Conv2D(filters=conv_filter_3, kernel_size=(3, 3), strides=(1, 1), padding="same")(x_2)
    x_2 = k.layers.Activation("relu")(x_2)

    x_3 = k.layers.Conv2D(filters=conv_filter_4, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x_3 = k.layers.Activation("relu")(x_3)
    x_3 = k.layers.Conv2D(filters=conv_filter_5, kernel_size=(5, 5), strides=(1, 1), padding="same")(x_3)
    x_3 = k.layers.Activation("relu")(x_3)

    x_4 = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    x_4 = k.layers.Conv2D(filters=conv_filter_6, kernel_size=(1, 1), strides=(1, 1), padding="same")(x_4)
    x_4 = k.layers.Activation("relu")(x_4)

    x = k.layers.Concatenate(axis=3)([x_1, x_2, x_3, x_4])

    return x


def googlemet_preoutput_module(x):
    x = k.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)

    x = k.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = k.layers.Activation("relu")(x)

    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=1024)(x)
    x = k.layers.Activation("relu")(x)

    return x


def output_module(x):
    x = k.layers.Dense(units=3)(x)
    x = k.layers.Activation("tanh")(x)

    return x


def googlenet_output_module(x):
    x = googlemet_preoutput_module(x)

    x = output_module(x)

    return x


def googlenet_input(x):
    # Up Sampling
    x = k.layers.UpSampling2D(size=(2, 2))(x)

    x = k.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(x)
    x = k.layers.Activation("relu")(x)

    x = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    x = k.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = k.layers.Activation("relu")(x)

    x = k.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = k.layers.Activation("relu")(x)

    x = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    x = googlenet_module(x, 64, 96, 128, 16, 32, 32)
    x = googlenet_module(x, 128, 128, 192, 32, 96, 64)

    x = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    x = googlenet_module(x, 192, 96, 208, 16, 48, 64)

    return x


def shallow_googlenet(x):
    x = googlenet_input(x)

    x = googlenet_output_module(x)

    return x


def googlenet(x):
    x = googlenet_input(x)

    x_1 = googlenet_output_module(x)

    x = googlenet_module(x, 160, 112, 224, 24, 64, 64)
    x = googlenet_module(x, 128, 128, 256, 24, 64, 64)
    x = googlenet_module(x, 112, 144, 288, 32, 64, 64)

    x_2 = googlenet_output_module(x)

    x = googlenet_module(x, 256, 160, 320, 32, 128, 128)

    x = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    x = googlenet_module(x, 256, 160, 320, 32, 128, 128)
    x = googlenet_module(x, 384, 192, 384, 48, 128, 128)

    x = googlemet_preoutput_module(x)
    
    return x, x_1, x_2


def identity_block(X, f, filters, stage, block):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = k.layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name =
                        conv_name_base + '2a', kernel_initializer = k.initializers.glorot_uniform(seed=0))(X)
    X = k.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = k.layers.Activation('relu')(X)

    # Second component of main path
    X = k.layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name =
                        conv_name_base + '2b', kernel_initializer = k.initializers.glorot_uniform(seed=0))(X)
    X = k.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = k.layers.Activation('relu')(X)

    # Third component of main path
    X = k.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1,1), padding='valid', name=
                        conv_name_base + '2c', kernel_initializer=k.initializers.glorot_uniform(seed=0))(X)
    X = k.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activatin
    X = k.layers.Add()([X, X_shortcut])
    X = k.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = k.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base +
'2a', kernel_initializer=k.initializers.glorot_uniform(seed=0))(X)
    X = k.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = k.layers.Activation('relu')(X)

    # Second component of main path
    X = k.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base +
'2b', kernel_initializer=k.initializers.glorot_uniform(seed=0))(X)
    X = k.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = k.layers.Activation('relu')(X)

    # Third component of main path
    X = k.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base +
'2c', kernel_initializer=k.initializers.glorot_uniform(seed=0))(X)
    X = k.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #####
    X_shortcut = k.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
name=conv_name_base + '1', kernel_initializer=k.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = k.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activatin
    X = k.layers.Add()([X, X_shortcut])
    X = k.layers.Activation('relu')(X)

    return X


def resnet(x):
    # Zero-Padding
    X = k.layers.ZeroPadding2D((3, 3))(x)

    # Stage 1
    X = k.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
                        kernel_initializer=k.initializers.glorot_uniform(seed=0))(X)
    X = k.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = k.layers.Activation('relu')(X)
    X = k.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = k.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    X = k.layers.Flatten()(X)

    return X


def fit_model(input_model, test_bool, save_bool, load_bool, apply_bool, input_path, input_prefix, output_path, epochs):
    if test_bool:
        print("Get random data")

        # random data for now
        x_train = np.random.rand(100, 100, 100, 2)  # 100 images, shape (100, 100), channels static & moving
        y_train = (np.random.rand(100, 3) * 2) - 1  # 100 3-vectors
    else:
        print("Get training data")

        x_train = get_x(input_path, input_prefix)
        y_train = get_y(input_path)

    if input_model is None:
        print("No input model")

        if load_bool:
            print("Load model from file")

            model = k.models.load_model(output_path + "/model.h5")
        else:
            print("Generate new model")

            input_x = k.layers.Input(x_train.shape[1:])

            x = resnet(input_x)

            x = output_module(x)

            model = k.Model(inputs=input_x, outputs=x)

            model.compile(optimizer=k.optimizers.Nadam(), loss=k.losses.mean_absolute_error)
    else:
        print("Using input model")

        model = input_model

    model.summary()

    print("Fitting model")

    model.fit(x_train, y_train, epochs=epochs, verbose=1)

    loss = model.evaluate(x_train, y_train, verbose=0)
    print("Train loss:", loss)

    print("Saving model")

    if save_bool:
        model.save(output_path + "/model.h5")

    if apply_bool:
        test_model(model, False, input_path, input_prefix, input_path, output_path)

    return model


def write_to_file(file, data):
    for i in range(len(data)):
        output_string = ""

        for j in range(len(data[i])):
            output_string = output_string + str(data[i][j]) + ","

        output_string = output_string[:-1] + "\n"

        file.write(output_string)


def test_model(input_model, test_bool, data_input_path, data_input_prefix, model_input_path, output_path):
    if test_bool:
        print("Get random data")

        # random data for now
        x_test = np.random.rand(100, 100, 100, 2)  # 100 images, shape (100, 100), channels static & moving
        y_test = np.random.rand(100, 3) * 2 - 1  # 100 3-vectors
    else:
        print("Get test data")

        x_test = get_x(data_input_path, data_input_prefix)
        y_test = get_y(data_input_path)

    if input_model is None:
        print("No input model")
        print("Load model from file")

        model = k.models.load_model(model_input_path + "/model.h5")
    else:
        model = input_model

    print("Applying model")

    output = model.predict(x_test)

    with open(output_path + "/output_transforms.csv", "w") as file:
        write_to_file(file, output)

    difference_matrix = output - y_test
    difference_vector = np.abs(difference_matrix.flatten())
    
    print("Max difference: " + str(difference_vector.max()))
    print("Mean difference: " + str(difference_vector.mean()))

    boolean_difference = []

    for i in range(len(output)):
        for j in range(len(output[i])):
            if output[i][j] - y_test[i][j] < 0.01:
                boolean_difference.append(np.array(0))
            else:
                boolean_difference.append(np.array(1))

    absolute_difference = sum(boolean_difference)

    print("Absolute boolean difference: " + str(absolute_difference) + "/" + str(len(y_test) * 3))
    print("Relative boolean difference: " + str(((absolute_difference / 3.0) / len(y_test)) * 100) + "%")

    with open(output_path + "/difference.csv", "w") as file:
        write_to_file(file, difference_matrix)


if __name__ == "__main__":
    fit_model_bool = True
    while_bool = False

    if fit_model_bool:
        while_model = None

        while True:
            print("Fit model")

            while_model = fit_model(while_model,
                                    False,
                                    False,
                                    while_bool,
                                    True,
                                    "../training_data/",
                                    ".nii",
                                    "../results/",
                                    1000)

            if not while_bool:
                break
    else:
        print("Test model")

        test_model(None, False, "../training_data/", ".nii", "../results/", "../results/")
