from ModelUtil import initializeModel, train
from TestUtil import test
import os
from tensorflow import keras as keras

if __name__ == "__main__":
    databasePath = "./database/train"
    savePath = "./catDogModel.h5"

    if not os.path.exists(savePath):
        model = initializeModel()
        train(model, databasePath, savePath)

    model = keras.models.load_model(savePath)
    test(model)