
from Progressbar import show_progressbar
from Sampler import Sampler
from PredictionModel import PredictionModel

"""
Implementation of N-way one-shot learning generation. 
"""
def n_way_one_shot_learning(clf: PredictionModel, count, path, n=20, verbose=False):
    prefix = str(n)+"-way one-shot learning: "
    if verbose: show_progressbar(i=0, max_i=count, prefix=prefix)
    # Do n-way one-shot learning "count" times.
    correct = 0
    for i in range(count):
        image_main, images, Y = Sampler.n_way_one_shot_learning(path=path, n=n)
        prediction = clf.predict_single_class(main=image_main, inputs=images)
        if prediction == Y:
            correct += 1
        if verbose: show_progressbar(i=i+1, max_i=count, prefix=prefix)
    if verbose: show_progressbar(i=count, max_i=count, prefix=prefix, finish=True)
    return correct*100/count