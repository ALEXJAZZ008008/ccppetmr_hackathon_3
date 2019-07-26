import matplotlib.pyplot as plt
import sirf.STIR as pet
import sirf.Reg as Reg
import random


def generate_image(initial_image):
    """
    Add a random number of ellipsis to an  initial image.
    """

    image_shape = initial_image.as_array().shape

    image = initial_image.clone()

    for i in range(random.randint(2,10)):
        shape = pet.EllipticCylinder()

        shape.set_length(1)
        shape.set_radii((random.uniform(1, image_shape[1]/8),
                         random.uniform(1, image_shape[2]/8)))

        radii = shape.get_radii()

        shape.set_origin((0,
                          random.uniform(-(image_shape[1]/4) + radii[1],
                                         image_shape[1]/4 - radii[1]),
                          random.uniform(-(image_shape[2]/4) + radii[0],
                                         image_shape[2]/4 - radii[0])))

        image.add_shape(shape, scale = random.uniform(0,1))

    return image

if __name__ == "__main__":
    initial_image = Reg.ImageData('blank_image.hv')
    print(initial_image.as_array().shape)
    image = generate_image(initial_image)

    plt.imshow(image.as_array()[0,:,:])
    plt.show()

