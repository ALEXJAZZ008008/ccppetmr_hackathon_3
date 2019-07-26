import sirf.STIR as pet
import keras_reg
from generate_data import generate_data

def main():
    initial_image = pet.ImageData('blank_image.hv')
    generate_data(initial_image, 100, 10)




if __name__ == "__main__":
    main()
