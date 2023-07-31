#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ## Activation Functions
# 
# This portion of the code goes over the plotting of various activation functions and the creation of figures used in Chapter 5.

# Here, we are creating the x data that we will pass through each activation function to get the resulting activations.

# In[9]:


x = np.linspace(-5, 5, 51)
x


# ### Sigmoid
# 
# 

# In[10]:


sig = lambda x: 1 / (1 + np.exp(-x))


# In[11]:


plt.title("Sigmoid Curve")
plt.ylabel("Sigmoid Output")
plt.xlabel("X Values")
plt.plot(x, sig(x))


# ### Tanh

# In[12]:


plt.title("Tanh Curve")
plt.ylabel("Tanh Output")
plt.xlabel("X Values")
plt.plot(x, np.tanh(x))


# ### Shifted Sigmoid
# 
# We can shift sigmoid down by 0.5 to make it zero-centered. 

# In[13]:


plt.plot(x, sig(x)-0.5)


# ### ReLU
# 

# In[14]:


relu = lambda x: np.maximum(0, x)


# In[15]:


plt.title("ReLU Curve")
plt.ylabel("ReLU Output")
plt.xlabel("X Values")
plt.plot(x, relu(x))


# In[ ]:




