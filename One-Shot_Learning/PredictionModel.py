
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
import numpy as np


class PredictionModel:
    """
    Class containing the logic for the predictive model.
    """

    clf = XGBRegressor(n_jobs=-1)

    """
    # Pipeline3
    # Score on the training set was:-0.2311721732
    clf = make_pipeline(
        StackingEstimator(
            estimator=ExtraTreesRegressor(
                max_features="log2",
                n_estimators=500,
                n_jobs=-1
            )
        ),
        RandomForestRegressor(
            max_features="sqrt",
            n_estimators=500,
            n_jobs=-1
        )
    )
    """



    def fit(self, X, Y):
        """
        Train the classifier.

        :param X: list of inputs.
        :param Y: list of targets.
        """
        self.clf.fit(np.array(X), np.array(Y))


    def predict(self, X):
        """
        Make Prediction using the classifier.

        :param X: list of inputs.
        :return: float between 0 and 1 representing the regression score.
        """
        return self.clf.predict(X)

    def get_model(self):
        """
        Return the prediction model.

        :return: classifier.
        """
        return self.clf



    def set_model(self, clf):
        """
        Update the prediction model.

        :param clf: new classifier.
        """
        self.clf = clf








"""
def custom_distance(x1, x2):
    a = PredictionModel.clf.predict([x1])
    b = PredictionModel.clf.predict([x2])
    return abs(a - b)


#clf2 = KNeighborsClassifier(n_neighbors=3, metric=custom_distance)


# Train the classifier (with KNN). 
def fit(self, X, Y):
    self.clf.fit(X[:int(len(X) / 2)], Y[:int(len(X) / 2)])
    self.clf2.fit(X[-int(len(X) / 2):], Y[-int(len(X) / 2):])

    
# Make Prediction using the classifier (with KNN). 
def predict(self, X):
    return self.clf2.predict(X)
"""
