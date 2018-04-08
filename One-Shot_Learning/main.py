
from TpotOptimization import tpot_optimization_reg
from TpotOptimization import tpot_optimization_clf
from NWayOneShotLearning import n_way_one_shot_learning
from PredictionModel import PredictionModel
import Sampler
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


def start_20_way_one_shot(clf):
    """
    Train the predictionModel and test it on 20-way one-shot learning
    """
    print("Start 20-way one-shot learning")
    score = n_way_one_shot_learning(clf=clf, count=100, dataset=Dataset.data_evaluation, n=20, verbose=True)
    print("Score =", score)



def start_cross_validation(count):
    """
    Run 5-fold cross validation on a given classifier.
    """
    clf = PredictionModel()
    print("Start cross validation")
    KFold_CV.kfold_cv_unique_datasets(
        clf=clf,
        train_path=Dataset.data_background,
        test_path=Dataset.data_evaluation,
        count=count,
        k_fold=5,
        verbose=True
    )


def pickle_models():
    """
    Generate pickle models.
    """
    ModelHandler.train_save_new_model(count=10000, verbose=True)
    ModelHandler.train_save_new_model(count=20000, verbose=True)
    ModelHandler.train_save_new_model(count=30000, verbose=True)
    ModelHandler.train_save_new_model(count=40000, verbose=True)
    ModelHandler.train_save_new_model(count=50000, verbose=True)


# TODO: Fix.
def save_illustration_images():
    """
    Generate and save images.
    :return:
    """
    image = Sampler._load_image(
        path="images_background",
        alphabet="Latin",
        character="character05",
        version="0687_05.png"
    )

    image = Sampler._raw_image_to_2D(image)
    [print(r) for r in image]
    print("")

    image = Sampler.extract_visual_object(image)
    [print(r) for r in image]
    print("")


def main():
    #start_tpot_optimazation(count=30000)

    #clf = ModelHandler.train_model(clf=PredictionModel(), count=1000, verbose=True)
    #start_20_way_one_shot(clf=clf)

    start_cross_validation(count=1000)


if __name__ == "__main__":
    main()