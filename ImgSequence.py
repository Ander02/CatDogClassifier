from skimage import io, transform
import tensorflow.keras as keras
import numpy as np
import math

class ImgSequence(keras.utils.Sequence):

    def __init__(self, files, batch_size, imageLength=50):
            self.files = files
            self.batch_size = batch_size
            self.imageLength = imageLength

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
       
        x = np.zeros([self.batch_size, self.imageLength, self.imageLength, 3])
        y = np.zeros([self.batch_size, 2])
        for i, _fileName in enumerate(batch_files):
            img = transform.resize(io.imread(_fileName), (self.imageLength, self.imageLength))
            cat = 1 if "cat" in _fileName else 0
            dog = 1 if "dog" in _fileName else 0
            x[i] = img
            y[i, 0] = cat
            y[i, 1] = dog

        return x, y