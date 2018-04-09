import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-0.1543347989725296
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(booster="dart", learning_rate=0.15, max_depth=4, n_estimators=100, n_jobs=-1, objective="reg:linear")),
    StackingEstimator(estimator=GradientBoostingRegressor(loss="lad", max_depth=3, n_estimators=25)),
    GradientBoostingRegressor(loss="ls", max_depth=3, n_estimators=10)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
