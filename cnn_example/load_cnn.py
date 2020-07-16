# -*- coding: utf-8 -*-
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
# import pandas as pd

# Load Model
model = load_model('my_model.h5')
model.load_weights('my_model_weights.h5')
Path = ["3.jpg", "6.jpg", "10.jpg"]

print(model.input_shape)
print(model.summary())
# img = image.load_img("0.jpg", 0)
for i in Path:
    
    img = cv2.imread(f"{i}", 0)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img.astype("float32")
    img_4 = img - np.amin(img)
    img_5 = 255 * img_4 / (np.amax(img_4))
    x_test_img = np.reshape(img_5, (1, 28, 28))
    x_Test4D = x_test_img.reshape(x_test_img.shape[0], 28, 28, 1).astype('float32')
    x_Test4D_normalize = (x_Test4D / np.amax(x_test_img))
    prediction = model.predict_classes(x_Test4D_normalize)

    print(prediction)