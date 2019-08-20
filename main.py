import sirf.STIR as PET
import keras_reg
from generate_data import generate_data
from correct_image import correct_data


def main(while_bool, generate_bool, fit_bool, test_bool, correct_bool):
    model = None

    while True:
        if generate_bool:
            print("Generate data")

            generate_data(PET.ImageData("blank_image.nii"),
                          1000,
                          1,
                          "/home/alex/Documents/SIRF-SuperBuild_install/bin/stir_math",
                          "/home/alex/Documents/SIRF-SuperBuild_install/bin/reg_resample")

        if fit_bool:
            print("Fit model")

            model = keras_reg.fit_model(model, False, True, True, False, "./training_data/", ".nii", "./results/", 1000)

        if test_bool:
            print("Test model")

            keras_reg.test_model(model, False,  "./training_data/", ".nii", "./results/", "./results/")

        if correct_bool:
            print("Correct model")

            correct_data("/home/alex/Documents/SIRF-SuperBuild_install/bin/reg_resample",
                         "./training_data/",
                         "./corrected_data/",
                         "./results/")

        if not while_bool:
            break


if __name__ == "__main__":
    main(True, True, True, True, True)
