
from xgboost import XGBRegressor
import numpy as np
import ImageHandler
import sys


class PredictionModelSymmetricXGBoost:
    """
    Class containing the logic for the predictive model.
    """


    # -0.15169617230917484 (My test)
    # -0.13400621878282912 (Tpot test)
    clf = XGBRegressor(
        booster="dart",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=300,
        n_jobs=-1,
        objective="reg:linear"
    )


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
            pred = self.__predict_sample(sample)
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
            img1 = self.__format_image(x[0])
            img2 = self.__format_image(x[1])
            samples = self.__symmetrical_samples(img1, img2, Y[i])
            for sample in samples:
                sample[0] += (ImageHandler.ensure_1D_image(
                    ImageHandler.image_differences_2D(img1, img2))
                )
                new_samples.append(sample)
        X = self.__column(new_samples, 0)
        Y = self.__column(new_samples, 1)
        return X, Y


    """
                     PRIVATE FUNCTIONS
    """


    def __format_image(self, image):
        """
        Format an image using simple object detection and simple image convolution.

        :param image: 1D list or 2D list - representing an image.
        :return: 1D list representing an image.
        """
        img_cropped = self.__object_cropp_scale(image)
        img_conv = self.__image_convolution(img_cropped)
        return ImageHandler.ensure_1D_image(img_conv)


    def __image_convolution(self, image):
        """
        Applies image convolution to an image.

        :param image: 1D list or 2D list - representing an image.
        :return: 2D list representing an image after the convolution process.
        """
        # Gaussian blur 3 × 3.
        kernel = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]
        return self.__custom_image_convolution(image=image, kernel=kernel, division=16)


    def __custom_image_convolution(self, image, kernel, division=1):
        """
        Applies image convolution to an image with custom kernel.

        :param image: 1D list or 2D list - representing an image.
        :return: 2D list representing an image after the convolution process.
        """
        image = ImageHandler.ensure_2D_image(image)

        image_conv = []
        for i in range(len(image)):
            image_conv_row = []
            for j in range(len(image[0])):
                value = 0
                for i2 in range(len(kernel)):
                    for j2 in range(len(kernel[0])):
                        try:
                            value += image[i + i2 - 1][j + j2 - 1] * kernel[i2][j2]
                        except:
                            pass
                image_conv_row.append(value / division)
            image_conv.append(image_conv_row)
        return image_conv


    def __predict_sample(self, X):
        """
        Make single prediction using the classifier.

        :param X: list of two images.
        :return: float representing the prediction.
        """
        X, _ = self._format_fit_inputs([X], [[0]])
        pred1 = self.clf.predict([X[0]])[0]
        pred2 = self.clf.predict([X[1]])[0]
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
        image1_2D_raw = ImageHandler.ensure_2D_image(image1)
        image1_2D = ImageHandler.extract_visual_object_2D(image1_2D_raw)
        return ImageHandler.ensure_1D_image(image1_2D)


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
