import sirf.STIR as pet
from keras_reg import keras_reg
from generate_data import generate_data

def main():
    initial_image = pet.ImageData('blank_image.hv')

    generate_data(initial_image, 100, 10)

    keras_reg.fit_model(False, False, True, "training_data/", ".nii", "results/")

if __name__ == "__main__":
    main()
