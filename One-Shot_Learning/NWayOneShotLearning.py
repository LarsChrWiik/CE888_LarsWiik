
from Progressbar import show_progressbar
from PredictionModel import PredictionModel
import Sampler
import random

# TODO: move to NWayOneShotLearning.
def predict_most_similar_image(clf, image_main, X):
    """
    Predict single class.

    :param main:
    :param inputs:
    :return:
    """
    # Predict using mirror input features.
    predictions = []
    for image in X:
        aaa = Sampler.object_cropping_sample()


        feature1 = image_main + image
        pred1 = clf.predict([feature1])[0]
        feature2 = image + image_main
        pred2 = clf.predict([feature2])[0]
        final_pred = (pred1 + pred2) / 2.0
        predictions.append(final_pred)

    # Find the most similar.
    min_indexes = []
    min_value = float("inf")
    for i, v in enumerate(predictions):
        if v < min_value:
            min_indexes = [i]
            min_value = v
        elif v == min_value:
            min_indexes.append(i)

    if len(min_indexes) > 1:
        rnd_index = random.choice(min_indexes)
        min_indexes = [rnd_index]

    predictions = [1 for _ in range(len(predictions))]
    predictions[min_indexes[0]] = 0

    return predictions


def n_way_one_shot_learning(clf, count, dataset, n, verbose=False):
    """
    Implementation of N-way one-shot learning generation.
    """
    prefix = str(n)+"-way one-shot learning: "
    if verbose: show_progressbar(i=0, max_i=count, prefix=prefix)

    # Do n-way one-shot learning "count" times.
    correct = 0
    for i in range(count):
        image_main, X, Y = Sampler.n_way_one_shot_learning(dataset=dataset, n=n)
        prediction = predict_most_similar_image(clf=clf, image_main=image_main, X=X)
        if prediction == Y: correct += 1
        if verbose: show_progressbar(i=i+1, max_i=count, prefix=prefix)
    if verbose: show_progressbar(i=count, max_i=count, prefix=prefix, finish=True)

    return (correct*100) / count
