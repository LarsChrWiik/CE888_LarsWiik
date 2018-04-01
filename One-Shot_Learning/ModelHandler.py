
from PredictionModel import PredictionModel
from Sampler import Sampler
import Dataset
import pickle


"""
Get the path for a model.
"""
def get_model_string(count):
    return "./models/trained_model" + str(count) + ".pkl"


def train_model(clf, count, verbose=False):
    if verbose: print("Generate training samples")
    X_train, Y_train = Sampler.get_samples(path=Dataset.data_background, count=count)
    if verbose: print("Fit")
    clf.fit(X_train, Y_train)
    return clf


"""
Train and save a model to disk. 
"""
def train_save_model(clf, count, verbose=False):
    clf = train_model(clf=clf, count=count, verbose=True)
    # Save the model to disk.
    path = get_model_string(count=count)
    pickle.dump(clf.get_model(), open(path, 'wb'))
    if verbose: print("model saved: " + path)


"""
Load and pre-trained model from disk. 
"""
def load_model(count, verbose=False):
    # Load the model from disk.
    path = get_model_string(count=count)
    loaded_model = pickle.load(open(path, 'rb'))
    clf = PredictionModel()
    clf.set_model(loaded_model)
    if verbose: print("Model loaded: " + path)
    return clf


"""
Continue training a model. 
"""
def train_save_new_model(count, verbose=False):
    clf = PredictionModel()
    train_save_model(clf=clf, count=count, verbose=verbose)
