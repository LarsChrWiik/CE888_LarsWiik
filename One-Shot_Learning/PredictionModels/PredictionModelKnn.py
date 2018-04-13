
"""
This is a test file and is not used in this project.
"""

from PredictionModels.PredictionModelSymmetricXGBoost import PredictionModelSymmetricXGBoost
from sklearn.neighbors import KNeighborsClassifier


def custom_distance(x1, x2):
    """
    Custom distance function for KNN.

    :param x1: 1D list representing an image.
    :param x2: 1D list representing an image.
    :return: float representing the distance between the images.
    """
    a = PredictionModelKnn.clf.predict([x1])
    b = PredictionModelKnn.clf.predict([x2])
    return abs(a - b)


class PredictionModelKnn(PredictionModelSymmetricXGBoost):
    """
    Class to handle Knn prediction using PredictionModel.
    """

    clf2 = KNeighborsClassifier(n_neighbors=1, metric=custom_distance)


    def fit(self, X, Y):
        """
        Train the classifier.

        :param X: list of image pars.
        :param Y: list of targets.
        """
        X, Y = self._format_fit_inputs(X, Y)
        self.clf.fit(X[:int(len(X) / 2)], Y[:int(len(Y) / 2)])
        self.clf2.fit(X[-int(len(X) / 2):], Y[-int(len(Y) / 2):])


    def predict(self, X):
        """
        Make prediction using the knn-classifier.

        :param X: list of image pars.
        :return: list of floats between 0 and 1 representing the regression score.
        """
        X, _ = self._format_fit_inputs(X, [[0] for _ in range(len(X))])
        X = [x for i, x in enumerate(X) if i % 2 == 0]
        return self.clf2.predict(X)
