import subprocess
import os
import shutil
import re
import math
import numpy as np
import sirf.Reg as Reg


def transform_image(moving_im_name, reg_resample, tr_x, tr_y, angle):
    """
    Randomly transform 2D image by translation or rotation.
    fixed_im_name   =
    """

    angle = angle * -1
    tr_x = tr_x * -1
    tr_y = tr_y * -1

    trans_file = 'temp_trans_file.txt'

    theta = angle * (math.pi / 2)

    transform = Reg.AffineTransformation(np.array(
        [[math.cos(theta), -math.sin(theta), 0, tr_x * 25],
         [math.sin(theta), math.cos(theta), 0, tr_y * 25],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]))

    transform.write(trans_file)

    args = [reg_resample,
            "-ref",
            moving_im_name,
            "-flo",
            moving_im_name,
            "-res",
            moving_im_name,
            "-trans",
            trans_file]
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    os.remove(trans_file)


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    return [atoi(c) for c in re.split(r"(\d+)", string)]


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
            x_fixed.append(relative_path + x_fixed_files[i])

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
                x_moving.append(temp_relative_path + x_moving_files_fixed_files[j])

        x_moving_fixed.append(x_moving)

    print("Got x moving")

    for i in range(len(x_moving_fixed)):
        for j in range(len(x_moving_fixed[i])):
            x.append(x_moving_fixed[i][j])

    print("Got x")

    return x


def get_y(output_input_path):
    print("Get y")

    y = []

    with open(output_input_path + "/output_transforms.csv", "r") as file:
        for line in file:
            line = line.rstrip()
            line_tuple = line.split(",")
            line_float = []

            for i in range(len(line_tuple)):
                line_float.append(float(line_tuple[i]))

            y.append(line_float)

    print("Got y")

    return np.nan_to_num(np.asarray(y))


def correct_data(reg_resample, move_path, input_path, output_input_path):
    if os.path.exists(input_path):
        shutil.rmtree(input_path)

    shutil.copytree(move_path, input_path)

    x = get_x(input_path, ".nii")
    y = get_y(output_input_path)

    for i in range(len(x)):
        transform_image(x[i], reg_resample, y[i][0], y[i][1], y[i][2])


def main():
    correct_data("/home/alex/Documents/SIRF-SuperBuild_install/bin/reg_resample",
                 "./training_data/",
                 "./corrected_data",
                 "./results/")


if __name__ == "__main__":
    main()
