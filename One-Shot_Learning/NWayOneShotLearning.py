
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

    correct = 0
    for i in range(count):
        image_main, X, Y = Sampler.n_way_one_shot_learning(dataset=dataset, n=n)
        prediction = predict(clf=clf, image_main=image_main, X=X)
        prediction_single = transform_to_signle_prediction(prediction)
        if prediction_single == Y: correct += 1

        # Save images for interpretability reasons.
        if interpretability and (prediction == Y and correct < 10) or (prediction != Y and i-correct < 10):
            __save_interpret_image(image=image_main, iteration=i, name="main_image", outside_images=True)
            for j, image in enumerate(X):
                __save_interpret_image(
                    image=image,
                    iteration=i,
                    name=str(round(prediction[j], 4)) + "_" + str(j)
                )
            __save_interpret_image(
                image=X[get_index_of_min_value(prediction)],
                iteration=i,
                name="main_image",
                outside_images=True
            )

        if verbose: show_progressbar(i=i+1, max_i=count, prefix=prefix)
    if verbose: show_progressbar(i=count, max_i=count, prefix=prefix, finish=True)

    return (correct*100) / count


def __remove_interpret_folder():
    rmtree("./interpretability", ignore_errors=True)


def __save_interpret_image(image, iteration, name, outside_images=False):
    img = ImageHandler.ensure_1D_image(image)
    size = int(math.sqrt(len(img)))
    img2 = Image.new("1", (size, size))
    img2.putdata(img)

    path = "./interpretability/" + str(iteration) + "/" + "images"
    if outside_images:
        path = "./interpretability/" + str(iteration)
    if not os.path.exists(path): os.makedirs(path)

    img2.save(path + "/" + name + '.png')