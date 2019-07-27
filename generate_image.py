import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sirf.STIR as PET
import random


def blur_image(input_image, sigma):
    im_array = input_image.as_array()

    input_image.fill(gaussian_filter(im_array, sigma))

    return input_image


def add_noise(input_image):
    im_array = input_image.as_array()

    input_image.fill(np.random.poisson(im_array / (np.sum(im_array) / (60000 * 120))))

    return input_image


def generate_image(input_image):
    """
    Add a random number of ellipsis to an  initial image.
    """

    image_shape = input_image.as_array().shape

    for i in range(random.randint(2, 10)):
        shape = PET.EllipticCylinder()

        shape.set_length(1)
        shape.set_radii((random.uniform(1, image_shape[1] / 8),
                         random.uniform(1, image_shape[2] / 8)))

        radii = shape.get_radii()
        shape.set_origin((0,
                          random.uniform(-(image_shape[1] / 4) + radii[1],
                                         image_shape[1] / 4 - radii[1]),
                          random.uniform(-(image_shape[2] / 4) + radii[0],
                                         image_shape[2] / 4 - radii[0])))

        input_image.add_shape(shape, scale=random.uniform(0, 1))

    input_image = add_noise(input_image)
    input_image = blur_image(input_image, 1)

    return input_image


if __name__ == "__main__":
    image = generate_image(PET.ImageData("blank_image.hv"))

    plt.imshow(image.as_array()[0, :, :])
    plt.show()
