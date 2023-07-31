#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# # Overfitting
# 
# In this notebook, we will be getting a neural network to overfit on some dataset to obtain the plots of the loss curves used in Chapter 5.
# 
# ### Loss Curve Example
# 
# Overfitting a simple CNN on the MNIST dataset and plotting the resulting loss curves

# In[465]:


from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[466]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[468]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[491]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input

input_layer = Input(shape=(28,28, 1))
layer1 = Conv2D(kernel_size=3, filters=32, strides=2, activation='relu')(input_layer)
layer2 = Conv2D(kernel_size=3, filters=16, strides=2, activation='relu')(layer1)
layer3 = Conv2D(kernel_size=3, filters=8, strides=2, activation='relu')(layer2)
flat = Flatten()(layer3)
output_layer = Dense(10, activation='softmax')(flat)

model = Model(input_layer, output_layer)
model.summary()


# In[492]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[493]:


history = model.fit(x_train, y_train, batch_size=128, validation_data=(x_val, y_val), epochs=64)


# In[494]:


history_dict = history.history


# Plotting the loss curve using the training history. It is easy to see the point where we have begun to overfit.

# In[495]:


plt.title("Loss Curve")

plt.plot(history_dict['loss'], label='training_loss')
plt.plot(history_dict['val_loss'], label='validation_loss')

plt.vlines(np.argmin(history_dict['val_loss']), 0, np.max(history_dict['loss']),
           label = 'overfitting', linestyles='dashed', color='red')

plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()


# In[496]:


print("Overfitting Boundary Epoch:", np.argmin(history_dict['val_loss']))


# In[497]:


plt.title("Accuracy Curve")

plt.plot(history_dict['accuracy'], label='training_accuracy')
plt.plot(history_dict['val_accuracy'], label='validation_accuracy')

plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()


# In[481]:


test_results = model.evaluate(x_test, y_test)


# In[482]:


test_results


# In[ ]:




