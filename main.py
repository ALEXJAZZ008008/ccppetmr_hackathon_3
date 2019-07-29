import sirf.STIR as PET
from keras_reg import keras_reg
from generate_data import generate_data


def main(while_bool):
    model = None

    while True:
        print("Generate data")

        generate_data(PET.ImageData("blank_image.nii"),
                      100,
                      1,
                      "/home/alex/Documents/SIRF-SuperBuild_install/bin/stir_math",
                      "/home/alex/Documents/SIRF-SuperBuild_install/bin/reg_resample")

        print("Fit model")

        model = keras_reg.fit_model(model, False, True, True, True, "./training_data/", ".nii", "./results/", 100)

        if not while_bool:
            break


if __name__ == "__main__":
    main(True)
