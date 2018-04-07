
from TpotOptimization import tpot_optimization_reg
from TpotOptimization import tpot_optimization_clf
from NWayOneShotLearning import n_way_one_shot_learning
from TestSampler import testSampler
from PredictionModel import PredictionModel
from Sampler import Sampler
import ModelHandler
import KFold_CV
import Dataset

import warnings
warnings.filterwarnings("ignore")


#TODO: check/fix
"""
Test cases. 
"""
def start_test_cases(list_of_paths):
    # Test Sample generation.
    for path in list_of_paths:
        testSampler(path)


"""
Starts TPOT optimization. 
"""
def start_tpot_optimazation(count):
    print("Start TPOT optimization with count =", count)
    tpot_optimization_reg(
        count=count,
        train_path=Dataset.data_background,
        test_path=Dataset.data_evaluation,
        verbose=True
    )


"""
Train the predictionModel and test it on 20-way one-shot learning
"""
def start_20_way_one_shot(clf):
    print("Start 20-way one-shot learning")
    score = n_way_one_shot_learning(clf=clf, count=100, path=Dataset.data_evaluation, n=20, verbose=True)
    print("Score =", score)


"""
Run 5-fold cross validation on a given classifier. 
"""
def start_cross_validation(count):
    clf = PredictionModel()
    print("Start cross validation")
    KFold_CV.custom_kfold_cross_validation(
        clf=clf,
        train_path=Dataset.data_background,
        test_path=Dataset.data_evaluation,
        count=count,
        k_fold=5,
        verbose=True
    )


"""
Generate pickle models. 
"""
def pickle_models():
    ModelHandler.train_save_new_model(count=10000, verbose=True)
    ModelHandler.train_save_new_model(count=20000, verbose=True)
    ModelHandler.train_save_new_model(count=30000, verbose=True)
    ModelHandler.train_save_new_model(count=40000, verbose=True)
    ModelHandler.train_save_new_model(count=50000, verbose=True)


"""
Main. 
"""
def main():
    #start_tpot_optimazation(count=30000)

    #pickle_models()
    #ModelHandler.train_save_new_model(count=1000, verbose=True)
    #clf = ModelHandler.load_model(count=1000, verbose=True)

    clf = ModelHandler.train_model(clf=PredictionModel(), count=50000, verbose=True)
    start_20_way_one_shot(clf=clf)

    #start_cross_validation(count=10000)



"""
Initial. 
"""
if __name__ == "__main__":
    # Run test cases.
    #test([data_background, data_evaluation])

    # Run main.
    main()


"""
# SCORES
pipeline (count=10 000):                    20-way one-shot learning score = 12.0 
pipeline (count=1 000):                     20-way one-shot learning score = 5.0 

pipeline2 (count=1 000):                    20-way one-shot learning score = 7.0 
pipeline2 (count=10 000):                   20-way one-shot learning score = 13.0 
pipeline2 (count=30 000):                   20-way one-shot learning score = 17.0  

pipeline3 (count=10 000):                   20-way one-shot learning score = 23.0 
pipeline3 (count=10 000) compressed:        20-way one-shot learning score = 18.0 
pipeline3 (count=10 000) compressed:        CV accuracy score = 0.62275 
pipeline3 (count=30 000):                   20-way one-shot learning score = 25.0 
pipeline3 (count=50 000) compressed:        20-way one-shot learning score = 25.0 

pipeline4 (count=10 000):                   20-way one-shot learning score = 14.0

pipeline5 (count=50 000) compressed:        20-way one-shot learning score = 14.0
 

RandomForestRegressor (count=10 000):       20-way one-shot learning score = 6.0 
RandomForestRegressor (count=1 000):        20-way one-shot learning score = 7.0 

RandomForestClassifier (count=10 000):      20-way one-shot learning score = 5.0 

GradientBoostingRegressor (count=10 000)    20-way one-shot learning score = 12.0 

XGBRegressor (count=1 000)                  20-way one-shot learning score = 10.0 
XGBRegressor (count=10 000)                 20-way one-shot learning score = 18.0 
XGBRegressor (count=30 000)                 20-way one-shot learning score = 12.0 




with knn:   10000 RandomForestRegressor n=1   = 12
with knn:   10000 RandomForestRegressor n=1  compressed  = 8

with knn:   10 000 XGBoost n=1 raw_image = 10





"""


