
from Progressbar import show_progressbar
import Sampler
import random
import ImageHandler
import math
from PIL import Image
import os
from shutil import rmtree


def get_index_of_min_value(a_list):
    """
    Calculate the index in a given array that contain the lowest value.

    :param a_list: list of values.
    :return: int representing the index that contain the lowest value.
    """
    min_indexes = []
    min_value = float("inf")
    for i, v in enumerate(a_list):
        if v < min_value:
            min_indexes = [i]
            min_value = v
        elif v == min_value:
            min_indexes.append(i)

    if len(min_indexes) > 1:
        rnd_index = random.choice(min_indexes)
        min_indexes = [rnd_index]

    return min_indexes[0]


def predict(clf, image_main, X):
    """
    Predict single class.

    :param clf: classifier.
    :param image_main: list representing an image.
    :param X: list of images.
    :return: list of float representing the prediction.
    """
    predictions = []
    for image in X:
        pred = clf.predict([[image_main, image]])[0]
        predictions.append(pred)
    return predictions


def transform_to_signle_prediction(predictions):
    """
    Transform a float prediction into a 1-class prediction.

    :param predictions: list of float representing the prediction.
    :return: list of int representing a single class prediction.
    """
    min_index = get_index_of_min_value(predictions)
    predictions = [1 for _ in range(len(predictions))]
    predictions[min_index] = 0
    return predictions


def n_way_one_shot_learning(clf, count, dataset, n, verbose=False, interpretability=False):
    """
    Implementation of N-way one-shot learning generation.

    :param clf: classifier.
    :param count: int representing training count.
    :param dataset: string of the dataset.
    :param n: number of other images in the n-way one shot test.
    :param verbose: bool representing whether information should be shown.
    :param interpretability: bool indicating if images should be saved to disk for interpretability reasons.
    :return: float representing the final score of the test.
    """
    prefix = str(n)+"-way one-shot learning: "
    if verbose: show_progressbar(i=0, max_i=count, prefix=prefix)

    if interpretability: __remove_interpret_folder()

    correct_count = 0
    for i in range(count):
        image_main, X, Y, alphabet = Sampler.n_way_one_shot_learning(dataset=dataset, n=n)
        prediction = predict(clf=clf, image_main=image_main, X=X)
        prediction_single = transform_to_signle_prediction(prediction)

        # Save images for interpretability reasons.
        __save_image_interpretability(
            image_main=image_main,
            prediction=prediction,
            prediction_single=prediction_single,
            X=X, Y=Y, i=i,
            correct_count=correct_count,
            alphabet=alphabet,
            interpretability=interpretability
        )

        if prediction_single == Y: correct_count += 1
        if verbose: show_progressbar(i=i+1, max_i=count, prefix=prefix)
    if verbose: show_progressbar(i=count, max_i=count, prefix=prefix, finish=True)

    return (correct_count*100) / count


def __save_image_interpretability(
        image_main,
        prediction,
        prediction_single,
        X,
        Y,
        i,
        correct_count,
        alphabet,
        interpretability
):
    """
    Function to save images during 20-way one-shot learning for interpretability reasons.

    :param image_main: list of float.
    :param prediction: list of float.
    :param prediction_single: lost of int.
    :param X: list of images.
    :param Y: list of targets.
    :param i: int, iteration number.
    :param correct_count: int, count of number of correct classifications.
    :param alphabet: string, name of the alphabet
    :param interpretability: bool indicating if interpretability should be considered.
    :return:
    """
    image_save_limit = 10
    if interpretability and ((prediction_single == Y and correct_count < image_save_limit)
                             or (prediction_single != Y and i - correct_count < image_save_limit)):
        correct_or_wrong = "correct" if prediction_single == Y else "wrong"
        folder_name = str(i + 1) + "_" + correct_or_wrong
        for j, image in enumerate(X):
            target_name = "__target" if j == get_index_of_min_value(Y) else ""
            __save_interpret_image(
                image=image,
                folder_name=folder_name,
                name=str("%.3f" % prediction[j]) + "__score" + str(j) + target_name
            )
        __save_interpret_image(
            image=image_main,
            folder_name=folder_name,
            name="main_image",
            outside_images=True
        )
        __save_interpret_image(
            image=X[get_index_of_min_value(prediction)],
            folder_name=folder_name,
            name=str(round(prediction[get_index_of_min_value(prediction)], 4)),
            outside_images=True
        )
        __save_interpret_alphabet(folder_name=folder_name, alphabet=alphabet)


def __save_interpret_alphabet(folder_name, alphabet):
    """
    Save alphabet name to txt file.

    :param folder_name: string, name of the iteration folder.
    :param alphabet: string, name of the alphabet.
    """
    with open("./interpretability/" + folder_name + "/alphabet.txt", "w") as text_file:
        text_file.write(alphabet)


def __remove_interpret_folder():
    """
    Remove old interpretability files.
    """
    rmtree("./interpretability", ignore_errors=True)


def __save_interpret_image(image, folder_name, name, outside_images=False):
    """
    Save image to disk for interpretability reasons.

    :param image: list, image to be saved.
    :param folder_name: string, name of the iteration folder.
    :param name: string, name of the file.
    :param outside_images: bool, representing if the file should be stored outside the image folder.
    """
    img = ImageHandler.ensure_1D_image(image)
    size = int(math.sqrt(len(img)))
    img2 = Image.new("1", (size, size))
    img2.putdata(img)

    path = "./interpretability/" + str(folder_name) + "/" + "images"
    if outside_images:
        path = "./interpretability/" + str(folder_name)
    if not os.path.exists(path): os.makedirs(path)

    img2.save(path + "/" + name + '.png')
