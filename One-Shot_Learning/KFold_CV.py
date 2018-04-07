
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from Sampler import Sampler
import numpy as np


"""
Last stage of cross validation. (Fit and Predict). 
"""
def __cv_fit_predict(clf, X_train, Y_train, X_test, Y_test, verbose=False):
    # Fit and predict.
    if verbose: print("Fit")
    clf.fit(X_train, Y_train)
    if verbose: print("Predict")
    predictions = clf.predict(X_test)
    predictions = np.around(predictions)

    # Calculate score.
    score = len([x for i, x in enumerate(predictions) if x == Y_test[i]]) / len(predictions)
    return score


"""
Classic K-Fold validation. 
"""
def kfold_cross_validation(clf, X, Y, k_fold=5, verbose=False):
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


"""
Custom K-Fold validation.
User must input two paths (train and test). 
"""
def custom_kfold_cross_validation(clf, train_path, test_path, count=100, k_fold=5, verbose=False):
    cv_scores = []
    for i in range(k_fold):
        # Generate samples.
        if verbose: print("Generate training samples")
        X_train, Y_train = Sampler.get_samples(path=train_path, count=count)
        if verbose: print("Generate testing samples")
        X_test, Y_test = Sampler.get_samples(path=test_path, count=count)

        score = __cv_fit_predict(
            clf=clf,
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test,
            verbose=verbose
        )
        if verbose: print("(" + str(i) + ") - cv: " + str(score))
        cv_scores.append(score)

    return cv_scores
