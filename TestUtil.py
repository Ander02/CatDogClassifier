import numpy as np
from skimage import io, transform as skiTransform
import os
import matplotlib.pyplot as plt

def test(model):

    testDataBasePath = "./database/test/test"

    allTestFiles = [os.path.join(testDataBasePath, _fileName) for _fileName in os.listdir(testDataBasePath)]

    for _file in np.random.choice(allTestFiles, 10, replace=False):
        img = np.expand_dims(processImage(_file, output_shape=[32, 32]), 0)
        plt.imshow(plt.imread(_file))
        pred = model.predict(img)
        plt.title("Cat {:.2f}%\n Dog {:.2f}%".format(pred[0][0] * 100, pred[0][1] * 100))
        plt.show()
    plt.close()

def processImage(imagePath, output_shape):
    return skiTransform.resize(io.imread(imagePath), output_shape, clip=False)