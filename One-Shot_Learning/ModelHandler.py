
from PredictionModel import PredictionModel
import Sampler
import Dataset
import pickle


def get_model_string(count):
    """
    Get the path for a model.
    """
    return "./models/trained_model" + str(count) + ".pkl"


def train_model(clf, count, verbose=False):
    """
    Train a model.

    :param clf: classifier in sklearn format.
    :param count: int representing training count.
    :param verbose: bool representing whether information should be shown.
    :return: the trained classifier object.
    """
    if verbose: print("Generate training samples")
    X_train, Y_train = Sampler.generate_samples(dataset=Dataset.data_background, count=count)
    if verbose: print("Fit")
    clf.fit(X_train, Y_train)
    return clf


def train_save_model(clf, count, verbose=False):
    """
    Train and save a model to disk.

    :param clf: PredictionModel.
    :param count: int representing training count.
    :param verbose: bool representing whether information should be shown.
    """
    clf = train_model(clf=clf, count=count, verbose=True)
    # Save the model to disk.
    path = get_model_string(count=count)
    pickle.dump(clf.get_model(), open(path, 'wb'))
    if verbose: print("model saved: " + path)


def load_model(count, verbose=False):
    """
    Load and pre-trained model from disk.

    :param count: int representing training count.
    :param verbose: bool representing whether information should be shown.
    :return: PredictionModel object containing a trained classifier.
    """
    # Load the model from disk.
    path = get_model_string(count=count)
    loaded_model = pickle.load(open(path, 'rb'))
    clf = PredictionModel()
    clf.set_model(loaded_model)
    if verbose: print("Model loaded: " + path)
    return clf


def train_save_new_model(count, verbose=False):
    """
    Continue training a model.

    :param count: int representing training count.
    :param verbose: bool representing whether information should be shown.
    """
    clf = PredictionModel()
    train_save_model(clf=clf, count=count, verbose=verbose)
