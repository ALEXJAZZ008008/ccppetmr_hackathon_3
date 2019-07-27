import sirf.STIR as PET
from keras_reg import keras_reg
from generate_data import generate_data


def main(while_bool):
    while while_bool:
        generate_data(PET.ImageData("blank_image.nii"),
                      10,
                      10,
                      "/home/alex/Documents/SIRF-SuperBuild_install/bin/stir_math",
                      "/home/alex/Documents/SIRF-SuperBuild_install/bin/reg_resample")

        keras_reg.fit_model(False, while_bool, True, "./training_data/", ".nii", "./results/", 1000)


if __name__ == "__main__":
    main(True)
