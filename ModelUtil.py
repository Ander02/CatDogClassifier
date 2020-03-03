import numpy as np
from skimage import io, transform as skiTransform
import os
from ImgSequence import ImgSequence
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from TestUtil import test

def initializeModel():
    model = keras.Sequential()
    # model.add(keras.layers.Conv2D(filters=96, kernel_size=(11, 11), padding="same", activation="relu", strides=(4,4)))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation="relu", strides=(1,1)))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding="same",activation="relu", strides=(1,1)))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation="relu",strides=(1,1)))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", strides=(1,1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=1024, activation="relu"))
    model.add(keras.layers.Dense(units=1024, activation="relu"))
    model.add(keras.layers.Dense(units=2, activation="sigmoid"))
    return model

def processImage(imagePath, output_shape):
    return skiTransform.resize(io.imread(imagePath), output_shape, clip=False)

def processAndSaveImage(newImagePath, imagePath, output_shape):
    io.imsave(newImagePath, processImage(imagePath, output_shape))

def train(model, databasePath, savePath, batch_size=16, epochs=300, optimizer="rmsprop", metrics=["accuracy"], **kwargs):
    catFiles = [os.path.join(databasePath + "/train", _fileName) for _fileName in os.listdir(databasePath + "/train") if "cat" in _fileName]
    dogFiles = [os.path.join(databasePath + "/train", _fileName) for _fileName in os.listdir(databasePath + "/train") if "dog" in _fileName]

    files = np.random.choice(catFiles[:5000] + dogFiles[:5000], size=16, replace=False)

    validationFiles = catFiles[5000:6000] + dogFiles[5000:6000]

    imgSample = io.imread(files[0])

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)

    history = model.fit(x=ImgSequence(files=files, batch_size=batch_size, imageLength=32), 
                                    epochs=epochs,
                                    shuffle=True,
                                    validation_data=ImgSequence(files=validationFiles, batch_size=batch_size, imageLength=32), 
                                    **kwargs)
    model.save(savePath)

    plt.plot(history.history["loss"], history.history["val_loss"]) 
    print(model.summary())


    