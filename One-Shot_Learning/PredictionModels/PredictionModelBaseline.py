
from xgboost import XGBRegressor
import ImageHandler
from PredictionModels.PredictionModel import PredictionModel


class PredictionModelBaseline(PredictionModel):
    """
    This class is used as a Baseline.
    """

    clf = XGBRegressor()

    def fit(self, X, Y):
        """
        Train the classifier.

        :param X: list of image pars.
        :param Y: list of targets.
        """
        X = self.__format_X(X)
        self.clf.fit(X, Y)

    def predict(self, X):
        """
        Predicting the distance between two images.

        :param X: list of image pairs.
        :return: list of float.
        """
        X = self.__format_X(X)
        return self.clf.predict(X)


    def __format_X(self, X):
        """
        Format inputs by cropping the image and scaling the object.
        This also generate symmetrical sample combination.

        :param X:
        :return:
        """
        X_new = []
        for i, x in enumerate(X):
            img1 = ImageHandler.ensure_1D_image(x[0])
            img2 = ImageHandler.ensure_1D_image(x[1])
            X_new.append(list(img1)+list(img2))
        return X_new
