import subprocess
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import sirf.STIR as pet
import sirf.Reg as Reg

from generate_image import generate_image


def transform_image(fixed_im_name, moving_im_name):
    """
    Randomly transform 2D image by translation or rotation.
    fixed_im_name   =
    """

    trans_file = 'temp_trans_file.txt'

    angle = random.uniform(0,1)
    tr_x = random.uniform(0,1)
    tr_y = random.uniform(0,1)

    theta = (angle-0.5) * (math.pi / 2)

    #print('angle: {}\ntr_x: {}\ntr_y: {}\ntheta: {}'.format(angle, tr_x, tr_y,
    #                                                        theta))

    transform = Reg.AffineTransformation(np.array(
        [[math.cos(theta),-math.sin(theta),0,(tr_x-0.5)*50],
         [math.sin(theta),math.cos(theta),0,(tr_y-0.5)*50],
         [0,0,1,0],
         [0,0,0,1]]))

    transform.write(trans_file)

#    args = ["reg_resample", "-ref", "training_data/fixed/fixed_000.nii",
#            "-flo", "training_data/fixed/fixed_000.nii", "-res", "test",
#            "-trans", "temp_trans_file.txt"]
#
#    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
#    popen.wait()
#

    args = ["reg_resample", "-ref", fixed_im_name + '.nii', "-flo",
            fixed_im_name + '.nii', "-res", moving_im_name + '.nii', "-trans",
            trans_file]
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    os.remove(trans_file)

    return [-tr_x, -tr_y, -angle]


def generate_data(initial_image, num_images, num_tranforms):
    """
    Generate data for image registration NN.

    initial_image   =   Initial image used to generate fixed and moving images.
    num_images      =   Number of fixed images to generate.
    num_tranforms   =   number and transforms (and miving images) to generate
                        for each fixed image.
    """
    transforms = np.zeros((num_images * num_tranforms, 3))

    with open('training_data/transforms.csv', 'w') as transform_csv:

        for i in range(num_images):
            fixed_image = generate_image(initial_image)

            fixed_im_name = 'training_data/fixed/fixed_{:03d}'.format(i)

            fixed_image.write(fixed_im_name)

            args = ("stir_math", "--output-format", "stir_ITK.par", fixed_im_name,
                 fixed_im_name + '.hv')
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()

            try:
                os.mkdir('training_data/moving/fixed_{:03d}'.format(i))
            except:
                pass

            for j in range(num_tranforms):

                moving_im_name = 'training_data/moving/fixed_{:03d}/moving_{:03d}'.format(i, j)

                transform = transform_image(fixed_im_name, moving_im_name)

                transform_csv.write(','.join(str(tr) for tr in transform) +
                                    '\n')


def main():

    initial_image = pet.ImageData('blank_image.hv')

    generate_data(initial_image, 10, 10)

if __name__ == "__main__":
    main()

