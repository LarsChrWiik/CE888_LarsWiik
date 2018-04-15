
from TpotOptimization import tpot_optimization_reg
from NWayOneShotLearning import n_way_one_shot_learning
from PredictionModels.PredictionModelSymmetricXGBoost import PredictionModelSymmetricXGBoost
from PredictionModels.PredictionModelHausdorff import PredictionModelHausdorff
from PredictionModels.PredictionModelBaseline import PredictionModelBaseline
import ModelHandler
import KFold_CV
import Dataset

import warnings
warnings.filterwarnings("ignore")


def start_tpot_optimazation(count):
    """
    Starts TPOT optimization.
    """
    print("Start TPOT optimization with count =", count)
    tpot_optimization_reg(
        count=count,
        train_path=Dataset.data_background,
        test_path=Dataset.data_evaluation,
        verbose=True
    )


def start_20_way_one_shot(clf, count, interpretability=False):
    """
    Train the predictionModel and test it on 20-way one-shot learning
    """
    print("Start 20-way one-shot learning")
    score = n_way_one_shot_learning(
        clf=clf,
        count=count,
        dataset=Dataset.data_evaluation,
        n=20,
        verbose=True,
        interpretability=interpretability
    )
    print("Score =", score)



def start_cross_validation(clf, count, k_fold):
    """
    Run 5-fold cross validation on a given classifier.
    """
    print("Start cross validation")
    KFold_CV.kfold_cv_unique_datasets(
        clf=clf,
        train_path=Dataset.data_background,
        test_path=Dataset.data_evaluation,
        count=count,
        k_fold=k_fold,
        verbose=True
    )


def pickle_models():
    """
    Generate pickle models.
    """
    ModelHandler.train_save_new_model(count=5000, verbose=True)
    ModelHandler.train_save_new_model(count=10000, verbose=True)
    ModelHandler.train_save_new_model(count=20000, verbose=True)
    ModelHandler.train_save_new_model(count=30000, verbose=True)


def main():
    """
    Main function.
    """
    #start_tpot_optimazation(count=10000)

    clf = PredictionModelSymmetricXGBoost()

    clf = ModelHandler.train_model(clf=clf, count=10000, verbose=True)
    start_20_way_one_shot(clf=clf, count=400)

    #start_cross_validation(clf=PredictionModel(), count=1000, k_fold=5)



if __name__ == "__main__":
    main()
