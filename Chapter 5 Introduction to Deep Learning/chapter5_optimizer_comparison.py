#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Nadam, SGD, RMSprop, Adadelta


# # Optimizer Comparison
# 
# Using TensorFlow and the MNIST dataset to create a quick comparison graph of the various optimizers

# In[15]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input

def get_model():
    input_layer = Input(shape=(28,28, 1))
    layer1 = Conv2D(kernel_size=3, filters=32, strides=2, activation='relu')(input_layer)
    layer2 = Conv2D(kernel_size=3, filters=16, strides=2, activation='relu')(layer1)
    layer3 = Conv2D(kernel_size=3, filters=8, strides=2, activation='relu')(layer2)
    flat = Flatten()(layer3)
    output_layer = Dense(10, activation='softmax')(flat)
    
    model = Model(input_layer, output_layer)
    return model


# In[16]:


results = {}


# ### SGD

# In[17]:


# SGD
model = get_model()
model.compile(optimizer=SGD(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=256, validation_data=(x_val, y_val), epochs=64)
results['sgd'] = history.history


# ### SGD + Momentum

# In[18]:


# SGD Momentum
model = get_model()
model.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=256, validation_data=(x_val, y_val), epochs=64)
results['sgd_momentum'] = history.history


# ### Nesterov Accelerated Gradient (NAG)
# 
# SGD + Nesterov

# In[19]:


# SGD Nesterov Momentum
model = get_model()
model.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=256, validation_data=(x_val, y_val), epochs=64)
results['sgd_nesterov'] = history.history


# ### RMSProp

# In[21]:


# RMSProp
model = get_model()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=256, validation_data=(x_val, y_val), epochs=64)
results['rmsprop'] = history.history


# ### Adam Optimizer

# In[22]:


# Adam
model = get_model()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=256, validation_data=(x_val, y_val), epochs=64)
results['adam'] = history.history


# ### Adam Optimizer + Nesterov Momentum (NAdam Optimizer)

# In[23]:


# NAdam
model = get_model()
model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=256, validation_data=(x_val, y_val), epochs=64)
results['nadam'] = history.history


# Plotting the training losses for all the optimizers

# In[29]:


plt.figure(figsize=(10,5))
for k in results.keys():
    plt.plot(results[k]['loss'], label=k)
plt.title("Comparison of Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# Plotting the validation losses for all the optimizers

# In[30]:


plt.figure(figsize=(10,5))
for k in results.keys():
    plt.plot(results[k]['val_loss'], label=k)
plt.title("Comparison of Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[ ]:




