#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Optimizers
# 
# Creating the true function that we want to apply various optimizers on. This function was selected because of the many "humps" or local minima that the optimizer must overcome given the initial starting position on the right side of the graph in order to reach the global minimum all the way to the left of the graph.

# In[6]:


fn1 = lambda x: (np.sin(4*np.pi*x)/(3.75*x)) + (x-1)**4


# In[7]:


xx = np.linspace(0.5, 2.5, 100)
y = fn1(xx)


# In[8]:


plt.plot(xx, y)
plt.ylabel("y")
plt.xlabel("x")


# We will be using PyTorch's optimizers to help us perform the optimization steps

# In[9]:


import math
import torch 
from torch.autograd import Variable
from torch.optim import SGD, Adadelta, RMSprop, Adam, AdamW


# In[10]:


cost = lambda x: (torch.sin(4 * math.pi * x) / (3.75 * x)) + (x-1)**4
history = {}


# ### Stochastic Gradient Descent (SGD)
# 
# SGD no momentum

# In[13]:


#### SGD no momentum

x = torch.tensor([2.25], requires_grad=True)

optim = SGD([x], lr=1e-2)

optim_name = 'sgd'
history[optim_name] = {'x': [], 'cost': []}

n = 1000

for f in range(n):
    optim.zero_grad()
    
    loss = cost(x)
    
    history[optim_name]['x'].append(x.detach().item())
    history[optim_name]['cost'].append(loss.item())
    
    loss.backward()
    optim.step()

plt.title(f"Optimizer: {optim_name}")
print(f"Lowest Loss in {np.argmin(history[optim_name]['cost'])} iterations")
plt.scatter(history[optim_name]['x'], history[optim_name]['cost'], color='red')
plt.plot(xx, y)


# ### Stochastic Gradient Descent with Momentum
# 
# SGD with momentum

# In[19]:


#### SGD momentum

x = torch.tensor([2.25], requires_grad=True)

optim = SGD([x], lr=1e-2, momentum=0.9)

optim_name = 'sgd_momentum'
history[optim_name] = {'x': [], 'cost': []}

n = 1000

for f in range(n):
    optim.zero_grad()
    
    loss = cost(x)
    
    history[optim_name]['x'].append(x.detach().item())
    history[optim_name]['cost'].append(loss.item())
    
    loss.backward()
    optim.step()

plt.title(f"Optimizer: {optim_name}")
print(f"Lowest Loss in {np.argmin(history[optim_name]['cost'])} iterations")
plt.scatter(history[optim_name]['x'], history[optim_name]['cost'], color='red')
plt.plot(xx, y)


# ### Stochastic Gradient Descent with Nesterov Momentum
# 
# Also known as Nesterov Accelerated Gradient (NAG)

# In[20]:


#### SGD nesterov momentum

x = torch.tensor([2.25], requires_grad=True)

optim = SGD([x], lr=1e-2, momentum=0.9, nesterov=True)

optim_name = 'sgd_nesterov'
history[optim_name] = {'x': [], 'cost': []}

n = 1000

for f in range(n):
    optim.zero_grad()
    
    loss = cost(x)
    
    history[optim_name]['x'].append(x.detach().item())
    history[optim_name]['cost'].append(loss.item())
    
    loss.backward()
    optim.step()

plt.title(f"Optimizer: {optim_name}")
print(f"Lowest Loss in {np.argmin(history[optim_name]['cost'])} iterations")
plt.scatter(history[optim_name]['x'], history[optim_name]['cost'], color='red')
plt.plot(xx, y)


# ### Adadelta
# 

# In[21]:


#### Adadelta

x = torch.tensor([2.25], requires_grad=True)

optim = Adadelta([x]) 

optim_name = 'adadelta'
history[optim_name] = {'x': [], 'cost': []}

n = 1000

for f in range(n):
    optim.zero_grad()
    
    loss = cost(x)
    
    history[optim_name]['x'].append(x.detach().item())
    history[optim_name]['cost'].append(loss.item())
    
    loss.backward()
    optim.step()

plt.title(f"Optimizer: {optim_name}")
print(f"Lowest Loss in {np.argmin(history[optim_name]['cost'])} iterations")
plt.scatter(history[optim_name]['x'], history[optim_name]['cost'], color='red')
plt.plot(xx, y)


# ### RMSProp

# In[22]:


#### RMSProp

x = torch.tensor([2.25], requires_grad=True)

optim = RMSprop([x]) 

optim_name = 'rms_prop'
history[optim_name] = {'x': [], 'cost': []}

n = 1000

for f in range(n):
    optim.zero_grad()
    
    loss = cost(x)
    
    history[optim_name]['x'].append(x.detach().item())
    history[optim_name]['cost'].append(loss.item())
    
    loss.backward()
    optim.step()

plt.title(f"Optimizer: {optim_name}")
print(f"Lowest Loss in {np.argmin(history[optim_name]['cost'])} iterations")
plt.scatter(history[optim_name]['x'], history[optim_name]['cost'], color='red')
plt.plot(xx, y)


# ### Adam Optimizer

# In[23]:


#### Adam

x = torch.tensor([2.25], requires_grad=True)

optim = Adam([x], lr=3e-1) 

optim_name = 'adam'
history[optim_name] = {'x': [], 'cost': []}

n = 1000

for f in range(n):
    optim.zero_grad()
    
    loss = cost(x)
    
    history[optim_name]['x'].append(x.detach().item())
    history[optim_name]['cost'].append(loss.item())
    
    loss.backward()
    optim.step()

plt.title(f"Optimizer: {optim_name}")
print(f"Lowest Loss in {np.argmin(history[optim_name]['cost'])} iterations")
plt.scatter(history[optim_name]['x'], history[optim_name]['cost'], color='red')
plt.plot(xx, y)


# In[ ]:




