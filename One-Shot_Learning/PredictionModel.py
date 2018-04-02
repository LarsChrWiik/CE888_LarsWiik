
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
import random

class PredictionModel:


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
    Train the classifier. 
    """
    def fit(self, X, Y):
        return self.clf.fit(X, Y)


    """
    Make Prediction using the classifier. 
    """
    def predict(self, X):
        return self.clf.predict(X)


    """
    Predict single class. 
    """
    # TODO: clean up.
    def predict_single_class(self, main, inputs):

        # Predict using mirror input features.
        predictions = []
        for element in inputs:
            feature1 = main + element
            pred1 = self.predict([feature1])[0]
            feature2 = element + main
            pred2 = self.predict([feature2])[0]
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


    """
    Return the prediction model. 
    """
    def get_model(self):
        return self.clf


    """
    Update the prediction model. 
    """
    def set_model(self, clf):
        self.clf = clf





# TODO: Remove.
    """
    # pipeline2
    # Score on the training set was:-0.2424072
    clf = make_pipeline(
        StackingEstimator(
            estimator=RandomForestRegressor(
                max_features="log2",
                n_estimators=10,
                n_jobs=-1
            )
        ),
        RandomForestRegressor(
            max_features="sqrt",
            n_estimators=100,
            n_jobs=-1
        )
    )
    """

    """
    # pipeline1
    # -0.2541114
    clf = make_pipeline(
        StackingEstimator(
            estimator=GradientBoostingRegressor(
                loss="ls",
                max_depth=3,
                n_estimators=100
            )
        ),
        RandomForestRegressor(
            max_features="log2",
            n_estimators=25,
            n_jobs=-1
        )
    )
    """
