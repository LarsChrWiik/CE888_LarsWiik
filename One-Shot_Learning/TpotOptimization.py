
from tpot import TPOTRegressor
import numpy as np

from Sampler import Sampler


"""
This function is used to optimize algorithms and parameters using TPOT. 

Current best:
    # Score on the training set was:-0.24953600000000004
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=GradientBoostingRegressor(loss="ls", max_depth=3, n_estimators=100)),
        RandomForestRegressor(max_features="log2", n_estimators=25, n_jobs=-1)
    )
"""
def tpot_optimization(count, train_path, test_path):
    # Generate samples.
    X_train, Y_train = Sampler.get_samples(path=train_path, count=count)
    X_test, Y_test = Sampler.get_samples(path=test_path, count=count)

    # Custom TPOT optimization algorithms and parameters.
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
            'eta': [0.02, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [2, 4, 6, 8, 10],
            'n_jobs': [-1],
            'objective': ["reg:linear", "multi:softmax", "multi:softprob"]
        }
    }

    tpot = TPOTRegressor(
        generations=10,
        population_size=30,
        verbosity=2,
        scoring="neg_mean_squared_error",
        config_dict=tpot_config  # {'sklearn.ensemble.RandomForestRegressor': {}}
    )

    tpot.fit(np.array(X_train), np.array(Y_train))
    print(tpot.score(np.array(X_test, dtype=np.float64), np.array(Y_test, dtype=np.float64)))
    tpot.export('tpot_pipeline.py')