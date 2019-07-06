# The coins database is downloaded from  https://www.kaggle.com/wanderdust/coin-images#coins.zip
import h5py
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from scipy import misc
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

def convert_sed_to_h5py(image_size, full_path_file_out, full_path_in, range_ids_from_to):

    with open('data/cat_to_name.json') as json_file:
        data = json.load(json_file)


        train_set_x_orig = np.empty([0,0])
        train_set_y_orig = np.array([])

        # dev means validation here
        dev_set_x = []
        dev_set_y = []

        test_set_x = []
        test_set_y = []

        for id in range(range_ids_from_to[0], range_ids_from_to[1]):
            train_set_y_orig += [id]
            full_path_images = \
                os.path.join(full_path_in, str(id))

            for filename in os.listdir(full_path_images):
                path_image = os.path.join(full_path_images, filename)
                image = misc.imread(path_image)
                image = resize(image, image_size)
                imgplot = plt.imshow(image)
                #plt.show()
                image_reshaped = np.reshape(image, [-1, 1])

                if len(train_set_x_orig) == 0:
                    train_set_x_orig = image_reshaped
                else:
                    train_set_x_orig = np.concatenate([train_set_x_orig, image_reshaped], axis=1)

                train_set_y_orig = np.append(train_set_y_orig, [id])

        hf = h5py.File(full_path_file_out, 'w')
        hf.create_dataset('train_set_x', data=train_set_x_orig)
        hf.create_dataset('train_set_y', data=train_set_y_orig)
        hf.close()

    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels




    np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig =


    """

convert_sed_to_h5py(
    image_size=[500, 500],
    full_path_file_out='/home/octavian/Octavian/ContinuedFractions/NeuralNetwork/datasets/coins/train_set.h5',
    full_path_in='/home/octavian/Octavian/ContinuedFractions/NeuralNetwork/datasets/coins/data/train/',
    range_ids_from_to=[206, 212])