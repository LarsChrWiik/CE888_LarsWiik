
from tpot import TPOTRegressor
from tpot import TPOTClassifier
import numpy as np

from Sampler import Sampler


"""
Optimize algorithms and parameters using TPOT for Regression trees. 
"""
def tpot_optimization_reg(count, train_path, test_path, verbose=False):

    # Generate samples.
    if verbose: print("Get train samples. ")
    X_train, Y_train = Sampler.get_samples(path=train_path, count=count)
    if verbose: print("Get test samples. ")
    X_test, Y_test = Sampler.get_samples(path=test_path, count=count)

    tpot_config = {
        'sklearn.ensemble.RandomForestRegressor': {
            'n_estimators': [10, 25, 100, 300, 1000],
            'max_features': ["auto", "sqrt", "log2"],
            'max_depth': [2, 3, 4],
            'n_jobs': [-1]
        },
        'sklearn.ensemble.ExtraTreesRegressor': {
            'n_estimators': [10, 25, 100, 300, 1000],
            'max_features': ["auto", "sqrt", "log2"],
            'max_depth': [2, 3, 4],
            'n_jobs': [-1]
        },
        'sklearn.ensemble.GradientBoostingRegressor': {
            'n_estimators': [10, 25, 100, 300, 1000],
            'loss': ["ls", "lad", "huber", "quantile"],
            'max_depth': [2, 3, 4]
        },
        'xgboost.XGBRegressor': {
            'n_estimators': [10, 25, 100, 300, 1000],
            'booster': ["gbtree", "gblinear", "dart"],
            "learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [2, 4, 6, 8, 10],
            'n_jobs': [-1],
            'objective': ["reg:linear", "multi:softmax", "multi:softprob"]
        }
    }

    tpot = TPOTRegressor(
        generations=5,
        population_size=15,
        verbosity=2,
        config_dict=tpot_config
    )

    tpot.fit(np.array(X_train), np.array(Y_train))
    print(tpot.score(np.array(X_test, dtype=np.float64), np.array(Y_test, dtype=np.float64)))
    tpot.export('tpot_pipeline_reg.py')



"""
Optimize algorithms and parameters using TPOT for Classification trees. 
"""
def tpot_optimization_clf(count, train_path, test_path, verbose=False):
    # Generate samples.
    if verbose: print("Get train samples. ")
    X_train, Y_train = Sampler.get_samples(path=train_path, count=count)
    if verbose: print("Get test samples. ")
    X_test, Y_test = Sampler.get_samples(path=test_path, count=count)


    tpot_config = {
        'xgboost.XGBClassifier': {
            'max_depth': [2, 3, 4, 5],
            "learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [10, 20, 30, 40, 50, 100, 500],
            'objective': ["reg:linear", "multi:softmax", "multi:softprob"],
            'booster': ["gbtree", "gblinear", "dart"],
            'n_jobs': [-1]
        },
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [10, 20, 30, 40, 50, 100, 500],
            'criterion': ["gini", "entropy"],
            'max_features': ["auto", "sqrt", "log2"],
            'max_depth': [2, 3, 4, 5],
            'n_jobs': [-1]
        }
    }

    if verbose: print("Start TPOT optimization. ")

    tpot = TPOTClassifier(
        generations=5,
        population_size=15,
        verbosity=2,
        config_dict=tpot_config
    )

    tpot.fit(np.array(X_train), np.array(Y_train))
    print(tpot.score(np.array(X_test, dtype=np.float64), np.array(Y_test, dtype=np.float64)))
    tpot.export('tpot_pipeline_clf.py')