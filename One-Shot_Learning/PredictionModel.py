
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
import numpy as np
import ImageHandler
import sys


class PredictionModel:
    """
    Class containing the logic for the predictive model.
    """

    clf = XGBRegressor(
        max_depth=8,
        learning_rate=0.02,
        n_estimators=1000,
        objective="reg:linear",
        booster='gbtree',
        n_jobs=-1
    )

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

        :param X: list of image pars.
        :param Y: list of targets.
        """
        self.__assert_input(X)
        X, Y = self._format_fit_inputs(X, Y)
        self.clf.fit(np.array(X), np.array(Y))


    def predict(self, X):
        """
        Make prediction using the classifier.

        :param X: list of image pars.
        :return: list of floats between 0 and 1 representing the regression score.
        """
        self.__assert_input(X)
        return self.predict_symmetric(X)


    def predict_symmetric(self, X):
        """
        Make prediction using symmetric prediction of images.

        :param X: list of image tuples.
        :return: list of floats representing the prediction.
        """
        predictions = []
        for sample in X:
            pred = self.__predict_single(sample)
            predictions.append(pred)
        return predictions


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
                    PROTECTED FUNCTIONS
    """


    def _format_fit_inputs(self, X, Y):
        """
        Format inputs by cropping the image and scaling the object.
        This also generate symmetrical sample combination.

        :param X:
        :param Y:
        :return:
        """
        new_samples = []
        for i, x in enumerate(X):
            img1 = self.__object_cropp_scale(x[0])
            img2 = self.__object_cropp_scale(x[1])
            samples = self.__symmetrical_samples(img1, img2, Y[i])
            for sample in samples:
                new_samples.append(sample)
        X = self.__column(new_samples, 0)
        Y = self.__column(new_samples, 1)
        return X, Y


    """
                     PRIVATE FUNCTIONS
    """


    def __predict_single(self, X):
        """
        Make single prediction using the classifier.

        :param X: list of two images.
        :return: float representing the prediction.
        """
        X, _ = self._format_fit_inputs([X], [[0]])
        pred1 = self.clf.predict(X[0])[0]
        pred2 = self.clf.predict(X[1])[0]
        return (pred1 + pred2) / 2.0


    def __assert_input(self, X):
        """
        Assert the input type. This ensures ease of use and testing.

        :param X: inputs from either fit or predict.
        """
        try:
            assert len(X[0]) == 2
        except:
            sys.exit("PredictionModel: Fit require inputs with two images.")


    def __object_cropp_scale(self, image1):
        """
        Merge two samples by calculating the differences between two samples.

        :param image1: 1D list representing an image.
        :return: 1D list representing an image that is cropped and scaled.
        """
        image1_2D_raw = ImageHandler.image_1D_to_2D(image1)
        image1_2D = ImageHandler.extract_visual_object_2D(image1_2D_raw)
        return ImageHandler.image_2D_to_1D(image1_2D)


    def __symmetrical_samples(self, image1, image2, Y):
        """
        Generate two symmetrical image samples.

        :param image1: 1D list representing an image.
        :param image2: 1D list representing an image.
        :param Y: int representing the target. (Whether or not the images are similar).
        :return: list of two symmetrical image samples.
        """
        samples = []
        X = image1 + image2
        X2 = image2 + image1
        sample = [X, Y]
        sample2 = [X2, Y]
        samples.append(sample)
        samples.append(sample2)
        return samples


    def __column(self, matrix, i):
        """
        Get column from matrix.

        :param matrix: 2D list
        :param i: int that represent the column index that should be extracted.
        :return: list representing the desired column from the matrix.
        """
        return [row[i] for row in matrix]
