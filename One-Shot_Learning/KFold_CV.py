
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
import Sampler
import numpy as np
from NWayOneShotLearning import transform_to_signle_prediction


def kfold_cv(clf, X, Y, k_fold=5, verbose=False):
    """
    Classic K-Fold validation.

    :param clf: classifier in sklearn format.
    :param X: inputs.
    :param Y: targets.
    :param k_fold: int representing the number of k-folds.
    :param verbose: bool representing whether information should be shown.
    :return: list containing the cross validations scores.
    """
    cv_scores = []
    for train, test in KFold(n_splits=k_fold).split(X):
        # split data
        X_train = np.array(X)[train]
        Y_train = np.array(Y)[train]
        X_test = np.array(X)[test]
        Y_test = np.array(Y)[test]

        score = __cv_fit_predict(clf=clf, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        if verbose:
            print(score)
        cv_scores.append(score)

    return cv_scores


def kfold_cv_unique_datasets(clf, train_path, test_path, count=100, k_fold=5, verbose=False):
    """
    Custom K-Fold validation using different datasets.

    :param clf: classifier in sklearn format.
    :param train_path: path to the dataset used for training.
    :param test_path: path to the dataset used for testing.
    :param count: int representing training count.
    :param k_fold: int representing number of k-folds.
    :param verbose: bool representing whether information should be shown.
    :return: list containing the cross validations scores.
    """
    cv_scores = []
    for i in range(k_fold):
        # Generate samples.
        if verbose: print("Generate training samples")
        X_train, Y_train = Sampler.generate_samples(dataset=train_path, count=count)
        if verbose: print("Generate testing samples")
        X_test, Y_test = Sampler.generate_samples(dataset=test_path, count=count)

        score = __cv_fit_predict(
            clf=clf,
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test,
            verbose=verbose
        )
        if verbose: print("(" + str(i+1) + ") - cv: " + str(score))
        cv_scores.append(score)

    if verbose: print("Avg cv score = " + str(sum(cv_scores)/len(cv_scores)))

    return cv_scores


def __cv_fit_predict(clf, X_train, Y_train, X_test, Y_test, verbose=False):
    """
    Last stage of cross validation. (Fit and Predict).

    :param clf: classifier in sklearn format.
    :param X_train: training inputs.
    :param Y_train: trianing targets.
    :param X_test: testing inputs.
    :param Y_test: testing targets.
    :param verbose: bool representing whether information should be shown.
    :return: float representing the cross validation score.
    """
    # Fit and predict.
    if verbose: print("Fit")
    clf.fit(X_train, Y_train)
    if verbose: print("Predict")
    predictions = clf.predict(X_test)

    # Normalize predictions.
    min_value = min(predictions)
    max_value = max(predictions)
    predictions = [(pred - min_value) / (max_value - min_value) for pred in predictions]

    # Predict either 0 or 1 according to threshold.
    predictions = [0 if x < 0.5 else 1 for x in predictions]

    # Calculate score.
    score = len([x for i, x in enumerate(predictions) if x == Y_test[i]]) / len(predictions)
    return score