import subprocess
import os
import shutil
import math
import random
import numpy as np
import sirf.STIR as PET
import sirf.Reg as Reg

from generate_image import generate_image


def transform_image(fixed_im_name, moving_im_name, reg_resample):
    """
    Randomly transform 2D image by translation or rotation.
    fixed_im_name   =
    """

    trans_file = 'temp_trans_file.txt'

    angle = random.uniform(-1, 1)
    tr_x = random.uniform(-1, 1)
    tr_y = random.uniform(-1, 1)

    theta = angle * (math.pi / 2)

    transform = Reg.AffineTransformation(np.array(
        [[math.cos(theta), -math.sin(theta), 0, tr_x * 25],
         [math.sin(theta), math.cos(theta), 0, tr_y * 25],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]))

    transform.write(trans_file)

    args = [reg_resample,
            "-ref",
            fixed_im_name + ".nii",
            "-flo",
            fixed_im_name + ".nii",
            "-res",
            moving_im_name + ".nii",
            "-trans",
            trans_file]
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    os.remove(trans_file)

    return [tr_x, tr_y, angle]


def generate_data(initial_image, num_images, num_transforms, stir_math, reg_resample):
    """
    Generate data for image registration NN.

    initial_image   =   Initial image used to generate fixed and moving images.
    num_images      =   Number of fixed images to generate.
    num_tranforms   =   number and transforms (and miving images) to generate
                        for each fixed image.
    """

    path = "./training_data"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    os.mkdir("./training_data/fixed")
    os.mkdir("./training_data/moving")

    with open("./training_data/transforms.csv", 'w') as transform_csv:
        for i in range(num_images):
            fixed_image = generate_image(initial_image)

            fixed_im_name = "./training_data/fixed/fixed_{:03d}".format(i)

            fixed_image.write(fixed_im_name)

            args = (stir_math,
                    "--output-format",
                    "stir_ITK.par",
                    fixed_im_name,
                    fixed_im_name + ".hv")
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()

            path = "training_data/moving/fixed_{:03d}".format(i)
            if not os.path.exists(path):
                os.mkdir(path)

            for j in range(num_transforms):
                moving_im_name = "training_data/moving/fixed_{:03d}/moving_{:03d}".format(i, j)

                transform = transform_image(fixed_im_name, moving_im_name, reg_resample)

                transform_csv.write(','.join(str(tr) for tr in transform) + '\n')


def main():
    generate_data(PET.ImageData("blank_image.hv"),
                  10,
                  10,
                  "/home/alex/Documents/SIRF-SuperBuild_install/bin/stir_math",
                  "/home/alex/Documents/SIRF-SuperBuild_install/bin/reg_resample")


if __name__ == "__main__":
    main()
