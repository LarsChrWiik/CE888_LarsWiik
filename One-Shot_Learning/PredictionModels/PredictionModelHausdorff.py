
from scipy.spatial.distance import directed_hausdorff
import ImageHandler
from PredictionModels.PredictionModel import PredictionModel


class PredictionModelHausdorff(PredictionModel):
    """
    Prediction class using the standard Hausdorff distance function.
    """

    def fit(self, X, Y):
        """
        No fitting is needed for this prediction model.

        :param X: list of image pars.
        :param Y: list of targets.
        """
        pass

    def predict(self, X):
        """
        Predicting the distance between two images.

        :param X: list of image pairs.
        :return: list of float.
        """
        predictions = []
        for image_pair in X:
            img1_2D = ImageHandler.ensure_2D_image(image_pair[0])
            img2_2D = ImageHandler.ensure_2D_image(image_pair[1])
            distance = directed_hausdorff(img1_2D, img2_2D)
            predictions.append(distance[0])
        return predictions
