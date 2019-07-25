import numpy as np
import matplotlib.pyplot as plt
import sirf.STIR as pet
import sirf.Reg as Reg

from generate_image import generate_image


def transform_image(fixed_image):

    return np.array((1,2,3)), fixed_image


def generate_data(initial_image, num_images, num_tranforms):

    transforms = np.zeros((num_images * num_tranforms, 3))

    for i in range(num_images):
        fixed_image = generate_image(initial_image)

        fixed_image.write('training_data/fixed/fixed_{:03d}'.format(i))

        for j in range(num_tranforms):
            transform, moving_image = transform_image(fixed_image)

            transforms[ ((i+1) * (j+1)) - 1 ] = transform


            moving_image.write(
                'training_data/moving/moving_{:03d}_fixed_{:03d}'.format(j, i))
    np.savetxt('training_data/transforms.csv', transforms, delimiter=',')

def main():
    initial_image = pet.ImageData('blank_image.hv')

    print(initial_image.as_array().shape)

    generate_data(initial_image, 10, 10)

if __name__ == "__main__":
    main()
