
from abc import ABC, abstractmethod


class PredictionModel(ABC):
    """
    Abstract class of a prediction model.
    """

    @abstractmethod
    def fit(self, X, Y):
        """
        Train the classifier.

        :param X: list of image pars.
        :param Y: list of targets.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicting the distance between two images.

        :param X: list of image pairs.
        :return: list of float.
        """
        pass
