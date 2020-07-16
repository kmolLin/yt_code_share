#!/usr/bin/env python
# coding: utf-8

# # How to use Google Colab
# #  

# ### MNIST, Convolutional Neural Network (CNN)
# 
# ### Step - 3
# 
# ### MNIST由手寫阿拉伯數字組成，包含60,000個訓練樣本和10,000個測試樣本。
# 
# ##### data from: https://keras.io/datasets/#mnist-database-of-handwritten-digits
# ##### code modified from: TensorFlow+Keras[深度學習]人工智慧實務應用 / 林大貴
# 

# # (1) Import the data from Keras

# In[ ]:


from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(3)
from keras.datasets import mnist


# In[ ]:


# read in the file
from numpy import load 

# data = load('mnist.npz')
# lst = data.files
# print(lst)
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()


# In[ ]:


# x_test_image  = data['x_test']     
# x_train_image = data['x_train']
# y_test_label  = data['y_test']
# y_train_label = data['y_train']

print(x_train_image.shape)
print(y_train_label.shape)
print(x_test_image.shape)
print(y_test_label.shape)


# # (2) View the first 10 images and labels

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(12,14)

for i in range(0,10):
    ax=plt.subplot(5,5,1+i)
    ax.imshow(x_train_image[i], cmap='binary')
    title= "label=" +str(y_train_label[i])
    ax.set_title(title,fontsize=10) 
    ax.set_xticks([]);ax.set_yticks([])        
plt.show()


# # (3) Convert 2-D image to nx28x28x1 array, normalize the numbers

# In[ ]:


# convert 2-D 28x28 image to 4-D nx28x28x1  array

x_Train4D=x_train_image.reshape(x_train_image.shape[0],28,28,1).astype('float32')
x_Test4D=x_test_image.reshape(x_test_image.shape[0],28,28,1).astype('float32')


# In[ ]:


# normalize the image numbers to 0~1

x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
print(x_Train4D_normalize.shape)
print(x_Test4D_normalize.shape)


# # (4) Convert label number to one-hot encoding

# In[ ]:


# convert label numbers to one-hot encoding

y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
print(y_TrainOneHot.shape)
print(y_TestOneHot.shape)


# # (5) Use a Convolutional Neural Network

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1), 
                 activation='relu'))


# In[ ]:


# Enable this cell in the second step

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# In[ ]:


model.add(Flatten())


# In[ ]:


# Enable this cell in the second step

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))


# In[ ]:


model.add(Dense(10,activation='softmax'))


# In[ ]:


print(model.summary())


# # (6) Model training

# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy']) 


# In[ ]:


train_history=model.fit(x=x_Train4D_normalize, 
                        y=y_TrainOneHot,validation_split=0.2, 
                        epochs=50, batch_size=300,verbose=2)


# # (7) Training history

# In[ ]:


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


show_train_history(train_history,'acc','val_acc')


# In[ ]:


show_train_history(train_history,'loss','val_loss')


# # (8) Accuracy

# In[ ]:


scores = model.evaluate(x_Test4D_normalize, y_TestOneHot)
print()
print('accuracy=',scores[1])


# # (9) Prediction

# In[ ]:


prediction=model.predict_classes(x_Test4D_normalize)


# In[ ]:


def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[ ]:


plot_images_labels_prediction(x_test_image,y_test_label,
                              prediction,idx=320)


# # (10) Confusion matrix

# In[ ]:


pd.crosstab(y_test_label,prediction,
            rownames=['label'],colnames=['predict'])


# In[ ]:


# save and load weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')


# In[ ]:


model.save('my_model.h5')  
del model  # deletes the existing model

from keras.models import load_model
model = load_model('my_model.h5')
model.summary()

