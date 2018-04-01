
from TpotOptimization import tpot_optimization
from NWayOneShotLearning import n_way_one_shot_learning
from TestSampler import testSampler
from PredictionModel import PredictionModel
from Sampler import Sampler
import ModelHandler
import KFold_CV
import Dataset


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
    tpot_optimization(
        count=count,
        train_path=Dataset.data_background,
        test_path=Dataset.data_evaluation
    )


"""
Train the predictionModel and test it on 20-way one-shot learning
"""
def start_20_way_one_shot(clf, count):
    print("Start 20-way one-shot learning")
    score = n_way_one_shot_learning(clf=clf, count=100, path=Dataset.data_evaluation, n=20, verbose=True)
    print("Score =", score, "with count =", count)


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


def test_keras():
    from keras.layers import Input, Conv2D, Activation, Lambda, merge, Dense, Flatten, MaxPooling2D
    from keras.models import Model, Sequential
    from keras.wrappers import scikit_learn

    print("Sampler")
    X, Y = Sampler.get_samples_2D(path=Dataset.data_background, count=100)

    input_shape = (105, 105, 1)
    Z = Input(input_shape)

    print("Convnet")
    convnet = Sequential()
    convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (7, 7), activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (4, 4), activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256, (4, 4), activation='relu'))
    convnet.add(Flatten())
    #x = Dense(1)(x)
    from sklearn.ensemble import RandomForestRegressor
    model2 = scikit_learn.KerasRegressor(build_fn=RandomForestRegressor(), sk_params={})
    convnet.add(model2)

    #print("Features")
    #features = convnet.call(Z)
    #print(type(features))
    #print(features)
    predictions = Activation("sigmoid")



    #model = Model(input=Z, output=features)
    #print(model)
    b = convnet.fit(X, Y)
    print(b)


    #convnet.add(Dense(4096, activation="sigmoid"))



"""
Main. 
"""
def main():

    test_keras()

    #ModelHandler.train_save_new_model(count=10000, verbose=True)
    #clf = ModelHandler.load_model(count=10000, verbose=True)

    #start_cross_validation(count=10000)
    #start_20_way_one_shot(clf=PredictionModel(), count=10000)
    #start_tpot_optimazation(count=30000)


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
pipeline3 (count=30 000):                   20-way one-shot learning score = 25.0 

RandomForestRegressor (count=10 000):       20-way one-shot learning score = 6.0 
RandomForestRegressor (count=1 000):        20-way one-shot learning score = 7.0 

RandomForestClassifier (count=10 000):      20-way one-shot learning score = 5.0 

GradientBoostingRegressor (count=10 000)    20-way one-shot learning score = 12.0 

XGBRegressor (count=1 000)                  20-way one-shot learning score = 10.0 
XGBRegressor (count=10 000)                 20-way one-shot learning score = 18.0 
XGBRegressor (count=30 000)                 20-way one-shot learning score = 12.0 
"""
