
from tpot import TPOTRegressor
from tpot import TPOTClassifier
from PredictionModels.PredictionModelSymmetricXGBoost import PredictionModelSymmetricXGBoost
import Sampler
import numpy as np


"""
Optimize algorithms and parameters using TPOT for Regression trees. 
"""
def tpot_optimization_reg(count, train_path, test_path, verbose=False):

    # Generate samples.
    formater = PredictionModelSymmetricXGBoost()

    if verbose: print("Get train samples. ")
    X_train, Y_train = Sampler.generate_samples(dataset=train_path, count=count)
    X_train, Y_train = formater._format_fit_inputs(X=X_train, Y=Y_train)
    if verbose: print("Get test samples. ")
    X_test, Y_test = Sampler.generate_samples(dataset=test_path, count=count)
    X_test, Y_test = formater._format_fit_inputs(X=X_test, Y=Y_test)

    tpot_config = {
        'sklearn.ensemble.RandomForestRegressor': {
            'n_estimators': [10, 25, 50, 75, 100, 300],
            'max_features': ["auto", "sqrt", "log2"],
            'max_depth': [2, 3, 4, 5, 6, 8, 10],
            'n_jobs': [-1]
        },
        'sklearn.ensemble.ExtraTreesRegressor': {
            'n_estimators': [10, 25, 50, 75, 100, 300],
            'max_features': ["auto", "sqrt", "log2"],
            'max_depth': [2, 3, 4, 5, 6, 8, 10],
            'n_jobs': [-1]
        },
        'sklearn.ensemble.GradientBoostingRegressor': {
            'n_estimators': [10, 25, 50, 75, 100, 300],
            "learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2],
            'loss': ["ls", "lad", "huber", "quantile"],
            'max_depth': [2, 3, 4, 5, 6, 8, 10]
        },
        'sklearn.ensemble.AdaBoostRegressor': {
            'base_estimator': ["DecisionTreeRegressor, GradientBoostingRegressor, RandomForestRegressor"],
            'n_estimators': [10, 25, 50, 75, 100, 300],
            "learning_rate": [0.6, 0.7, 0.8, 0.9, 1.0],
            'loss': ["linear", "square", "exponential"],
            'max_depth': [2, 3, 4, 5, 6, 8, 10]
        },
        'xgboost.XGBRegressor': {
            'n_estimators': [10, 25, 50, 75, 100, 300],
            'booster': ["gbtree", "gblinear", "dart"],
            "learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [2, 3, 4, 5, 6, 8, 10],
            'n_jobs': [-1],
            'objective': ["reg:linear"]
        }
    }

    tpot = TPOTRegressor(
        generations=10,
        population_size=30,
        verbosity=2,
        config_dict=tpot_config,
        cv=5
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
    X_train, Y_train = Sampler.generate_samples(dataset=train_path, count=count)
    if verbose: print("Get test samples. ")
    X_test, Y_test = Sampler.generate_samples(dataset=test_path, count=count)


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