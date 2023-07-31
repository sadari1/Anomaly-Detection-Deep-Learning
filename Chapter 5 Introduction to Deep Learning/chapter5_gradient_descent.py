#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Gradient Descent
# 
# This notebook will go over how gradient descent works and set up the plots that were used in Chapter 5.

# We are plotting the "loss curve" represented by f(x) = x^2
# and its derivative + instantaneous slope at x=2

# In[2]:


f_x = lambda x: x**2
df_dx = lambda x: 4*x - 4
x = np.linspace(-3, 3)
y = f_x(x)

dy_dx = df_dx(np.linspace(1 ,3))

plt.title("Graph of x^2 and its Derivative at x=2")
plt.plot(x, y, label='y = x^2')
plt.plot(np.linspace(1 ,3), dy_dx, label='dy_dx = 2x')
plt.legend()


# We are doing the same, but for a more complicated function.

# In[4]:


f_x = lambda x: x**3 - 5*x**2
df_dx = lambda x: -8*x + 4
x = np.linspace(-3, 3)
y = f_x(x)

dy_dx = df_dx(np.linspace(1 ,3))

plt.title("Graph of x^3 - 5x^2 and its Derivative at x=2")
plt.plot(x, y, label='y = x^3 - 5x^2')
plt.plot(np.linspace(1 ,3), dy_dx, label='dy_dx = 3x^2 - 10x')
plt.legend()


# We are setting up the three-dimensional example of a loss-surface using the function f(x, z) = x^2 + z^2

# In[6]:


f_x_z = lambda x, z: x**2 + z**2

x = np.linspace(-1, 1)
z = np.linspace(-1, 1)

xx, zz = np.meshgrid(x, z)
y = xx**2 + zz**2


# We are plotting the loss surface as well as the gradient plus the tangent plane.

# In[7]:


fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(projection='3d')
df_dx = lambda x: 2*x
df_dz = lambda z: 2*z 

x = 0.75
z = 0.75

dfxz = df_dx(x) * (xx - x) + df_dz(z) * (zz - z) + (x**2 + z**2)

ax.view_init(elev=15, azim=-5)
ax.scatter([x], [z], [x**2 + z**2], color='red', s=200)
ax.plot_surface(xx, zz, y, label='y=x^2 + z^2')
ax.plot_surface(xx, zz, dfxz, color='orange')

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')


# ### Gradient Descent Example
# 
# With that out of the way, we will now try and approximate y=4x+3 starting with a different slope and intercept just using gradient descent.

# In[12]:


from sklearn.metrics import mean_squared_error

# true:
fn1 = lambda x: 4*x + 3

# randomly intialize m and b
m = 1
b = 0


# In[13]:


# Take four points to optimize m and b on
x = np.array([0, 1, 2, 3])
y = np.array([fn1(f) for f in x])

x, y


# Our initial predictions. Clearly, they're far off from what the y array is showing, so m and b are not optimal.

# In[11]:


preds = np.array([m*f + b for f in x])
preds


# Plotting the loss curve over all combinations of m and b. The goal is to discover the best m and b values that will minimize the loss.
# 
# We can visualize the loss curve to get a sense of where we are starting with the initial m, b values on the loss curve and what path the optimizer must take to get to the optimal m, b.

# In[14]:


m_grid = np.linspace(0, 5)
b_grid = np.linspace(0, 5)

mm, bb = np.meshgrid(m_grid, b_grid)

pairs = np.asarray(list(zip(mm.reshape(-1), bb.reshape(-1))))

mses = []
for f in range(len(pairs)):
    m, b = pairs[f]
    preds = [g*m + b for g in x]
    
    mse_val = mean_squared_error(y, preds)

    mses.append(mse_val)
mses = np.array(mses)


# In[15]:



fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(projection='3d')

ax.view_init(elev=10, azim=-55)
ax.set_box_aspect(aspect=None, zoom=0.8)

ax.scatter(pairs[:, 0], pairs[:, 1], mses.reshape(-1, 1), alpha=0.3)

ax.scatter([4], [3], [0], s=200, color='red')
ax.text(4, 3, -15, 'Optimal')

m = 1
b = 0

preds = [g*m + b for g in x]
mse_val = mean_squared_error(y, preds)
ax.scatter([m], [b], [mse_val], s=100, color='red')
ax.text(0.4, 0, 55, 'Initial')

ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('Loss')


# In[16]:


def mse_grad(y_true, y_pred):
    # We will perform the averaging in a later step instead of right here
    grad = -2 * (y_true-y_pred)
    return grad


# In[17]:


# Initial settings

m = 1 
b = 0

preds = [m * g + b for g in x]

plt.title(f"Estimated Line for m={m} and b={b}")
plt.plot(x, y)
plt.plot(x, preds)
plt.legend(["True Line", "Predicted Line"])


# Going 1000 iterations with a step size of 0.001

# In[380]:


m = 1
b = 0

step_size = 0.001
for f in range(1000):
    preds = np.array([m*g + b for g in x])
    loss = mean_squared_error(y,  preds)
    print(f'\r{f}: {loss}', end='')
    
    dL_dyhat = mse_grad(y, preds)
    dyhat_dm = x
    dyhat_db = np.ones(len(x))

    dL_dm = np.mean(dL_dyhat * dyhat_dm)
    dL_db = np.mean(dL_dyhat * dyhat_db)
    
    m = m - step_size * dL_dm
    b = b - step_size * dL_db


# In[381]:


preds = np.array([m*g + b for g in x])
preds


# In[382]:


m, b


# In[383]:


preds, y


# In[385]:


plt.title(f"Estimated Line for m={m} and b={b}")
plt.plot(x, y)
plt.plot(x, preds)
plt.legend(["True Line", "Predicted Line"])


# Replotting the loss curve but with the approximated m, b values to find out how much closer we are to optimal than the initial starting state.

# In[390]:



fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(projection='3d')

ax.view_init(elev=10, azim=-55)
ax.set_box_aspect(aspect=None, zoom=0.8)

ax.scatter(pairs[:, 0], pairs[:, 1], mses.reshape(-1, 1), alpha=0.3)

ax.scatter([4], [3], [0], s=200, color='red')
ax.text(4, 3, -15, 'Optimal')

mse_val = mean_squared_error(y, preds)
ax.scatter([m], [b], [mse_val], s=100, color='red')
ax.text(4, 2, -15, 'Estimated')

preds_original = [g*1 + 0 for g in x]
mse_val = mean_squared_error(y, preds_original)
ax.scatter([1], [0], [mse_val], s=100, color='red')
ax.text(0.4, 0, 55, 'Initial')

ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

