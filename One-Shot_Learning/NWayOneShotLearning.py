
from Progressbar import show_progressbar
from PredictionModel import PredictionModel
import Sampler
import random


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


def predict_most_similar_image(clf, image_main, X):
    """
    Predict single class.

    :param clf: classifier.
    :param image_main: list representing an image.
    :param X: list of images.
    :return: list of numbers representing a single class prediction.
    """
    predictions = []
    for image in X:
        pred = clf.predict([[image_main, image]])[0]
        predictions.append(pred)
    min_index = get_index_of_min_value(predictions)
    predictions = [1 for _ in range(len(predictions))]
    predictions[min_index] = 0
    return predictions


def n_way_one_shot_learning(clf, count, dataset, n, verbose=False):
    """
    Implementation of N-way one-shot learning generation.

    :param clf: classifier.
    :param count: int representing training count.
    :param dataset: string of the dataset.
    :param n: number of other images in the n-way one shot test.
    :param verbose:
    :return: float representing the final score of the test.
    """
    prefix = str(n)+"-way one-shot learning: "
    if verbose: show_progressbar(i=0, max_i=count, prefix=prefix)

    correct = 0
    for i in range(count):
        image_main, X, Y = Sampler.n_way_one_shot_learning(dataset=dataset, n=n)
        prediction = predict_most_similar_image(clf=clf, image_main=image_main, X=X)
        if prediction == Y: correct += 1
        if verbose: show_progressbar(i=i+1, max_i=count, prefix=prefix)
    if verbose: show_progressbar(i=count, max_i=count, prefix=prefix, finish=True)

    return (correct*100) / count
