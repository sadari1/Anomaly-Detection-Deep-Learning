#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Preparation
# 
# First, we must load the data and then transform it into a usable condition to pass into the One-Class Support Vector Machine (OC-SVM) algorithm.
# 
# We define a list of columns that correspond with the data. Unlike some other data files, the data does not contain columns so we must define them ourselves to tell pandas what the column labels should be. Otherwise, they'll be numbered by numerical index, making it harder to interpret the data.

# In[2]:


columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", 
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

df = pd.read_csv("data/kddcup.data.corrected", sep=",", names=columns, index_col=None)


# This is a very large dataset. Let's only concern ourselves with http attacks

# In[3]:


# Filter to only 'http' attacks
df = df[df["service"] == "http"]
df = df.drop("service", axis=1)


# There are many types of anomalous attacks. To make things simple, we will only detect whether a data point is normal or if it is an anomaly

# In[4]:


# Label of 'normal.' becomes 0, and anything else becomes 1 and is treated as an anomaly.
df['label'] = df['label'].apply(lambda x: 0 if x=='normal.' else 1)
df['label'].value_counts()


# The next step we should perform is to encode the categorical columns. However, there are many columns to search over, so let's automate this. The following cell will create a dictionary of column names to their corresponding data type. From that data type dictionary, we can iterate and find only the columns that are strings, which carry a datatype of "object" in pandas.
# 
# Additionally, we will be applying **standard scaling** to all the numeric columns. When all the variables are on a similar scale, the one-class support vector machine can arrive at a solution much more easily. This is a crucial step for support vector machine models.

# In[5]:


datatypes = dict(zip(df.dtypes.index, df.dtypes))
encoder_map = {}
for col, datatype in datatypes.items():
    if datatype == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoder_map[col] = encoder 
    else:
        if col == 'label':
            continue 
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        encoder_map[col] = scaler 


# In[6]:


# Check the variables with highest correlation with 'label'
df2 = df.copy()
label_corr = df2.corr()['label']


# Now that we have filtered out only the columns with a stronger correlation with label, let's split our data into train-test-val splits.

# In[7]:


# Filter out anything that has null entry or is not weakly correlated
train_cols = label_corr[(~label_corr.isna()) & (np.abs(label_corr) > 0.2)]
train_cols = list(train_cols[:-1].index)
labels = df2['label']

# Conduct a train-test split    
x_train, x_test, y_train, y_test = train_test_split(df2[train_cols].values, labels.values, test_size = 0.15, random_state = 42)


# In[8]:


# Additional split of training dataset to create validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[9]:


print("Shapes")
print(f"x_train:{x_train.shape}\ny_train:{y_train.shape}")
print(f"\nx_val:{x_val.shape}\ny_val:{y_val.shape}")
print(f"\nx_test:{x_test.shape}\ny_test:{y_test.shape}")


# Since the OC-SVM only trains on data belonging to one class (since it is used for novelty / outlier detection), we need to split the dataset into one part comprised only of normal instances (to be used for training), and another part that combines a mixture of normal data and just the anomalies.

# In[10]:


# Split a set into 80% only normal data, and 20% normal data + any anomalies in set
def split_by_class(x, y):
    # Separate into normal, anomaly
    x_normal = x[y == 0]
    x_anom = x[y==1]
    
    y_normal = y[y==0]
    y_anom = y[y==1]
    
    # Split normal into 80-20 split, one for pure training and other for eval
    x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_normal, y_normal, test_size=0.2, random_state=42)
    
    # Combine the eval set with the anomalies to test outlier detection
    x_train_test = np.concatenate((x_train_test, x_anom))
    y_train_test = np.concatenate((y_train_test, y_anom))
    
    # Shuffle the eval set
    random_indices = np.random.choice(list(range(len(x_train_test))), size=len(x_train_test), replace=False)
    x_train_test = x_train_test[random_indices]
    y_train_test = y_train_test[random_indices]
    
    return x_train_train, x_train_test, y_train_train, y_train_test
                        
    


# We will split up our training dataset this way.

# In[11]:


### Train on normal data only. The _test splits have normal and anomaly data both
x_train_train, x_train_test, y_train_train, y_train_test = split_by_class(x_train, y_train)


# In[12]:


print(f"x_train_train: {x_train_train.shape}")
print(f"y_train_train: {y_train_train.shape}")   
print(f"x_train_test: {x_train_test.shape}") 
print(f"y_train_test: {y_train_test.shape}") 


# One issue with the OC-SVM is that the training time scales very poorly with data. And so, for the sake of prototyping, we will be using far fewer samples than usual to find the optimal set of hyperparameters and then scale it back up. 

# In[13]:


# nu is a cap on the upper bound of training errors and lower bound of the fraction of support vectors. We will first try to enter the expected amount of anomalies
svm = OneClassSVM(nu=0.0065, gamma=0.05)
svm.fit(x_train_train[:50000])


# Predictions returned are either:
# - 1 -> normal data (inlier)
# - -1 -> anomaly data (outlier)
# 
# So, to convert the predictions to be in terms of 0 (normal) or 1 (anomaly), we only need to apply a boolean condition of:
# 
# preds < 0
# 
# Since -1, the anomaly, is less than 0, it evaluates to True, which is converted to 1.
# Likewise, 1, the normal point, is greater than 0, which evaluates to False, which converts to 0.
# 

# In[14]:


preds = svm.predict(x_train_test)
# -1 is < 0, so it flags as 1. 1 is > 0, so flags as 0.
preds = (preds < 0).astype(int) 
preds


# In[15]:


pre = precision_score(y_train_test, preds )
rec = recall_score(y_train_test, preds)
f1 = f1_score(y_train_test, preds)

print(f"Precision: {pre}")
print(f"Recall: {rec}")
print(f"F1-Measure: {f1}")


# This looks like a really good score just evaluating on a \_test split derived from the training data. 

# In[16]:


ConfusionMatrixDisplay(confusion_matrix(y_train_test, preds)).plot()


# The confusion matrix looks good overall - there were no anomalies that were false negatives, but there were more than a few false positives. Now let's see how this works on our original test split.

# In[17]:



preds = svm.predict(x_test)
preds = (preds < 0).astype(int) # -1 is < 0, so it flags as 1. 1 is > 0, so flags as 0.
preds


# In[18]:


pre = precision_score(y_test, preds )
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Precision: {pre}")
print(f"Recall: {rec}")
print(f"F1-Measure: {f1}")


# It seems like our model massively overfit to the training data, since the F1-measure is so low compared to earlier. It makes sense that the performance looked good on the x_train_test split because this was derived from the training data, and thus has a similar distribution. The test split was completely different and was never involved in the training process.
# 
# We now need to look at hyperparameter tuning to see if we can get this model to generalize better.
# 
# Of interest are the following hyperparameter settings:
# - **gamma**: A parameter used in the RBF kernel function. It determines the radius of influence that points have on each other, with smaller gamma translating to larger radii. We want the radius of influence to be as large as possible to include all the outliers, but not too large, or else the model ends up learning nothing important. If gamma is too large, then the SVM will try and fit the training data more closely, which might lead to overfitting. This makes gamma an important parameter to tune.
# 
# - **nu**: This is an upper bound on the fraction of training errors allowed and a lower bound on the fraction of support vectors. It is a more sensitive hyperparameter than gamma and it can influence the output behavior of the model quite a bit, so it is very important to tune this value according to your dataset. You may also need to retune this even if your training data size changes. A very small nu places a stricter requirement on how many training errors are allowed, while a larger nu value allows for more training errors to slip through when training the model.
# 

# In[19]:


# Given an svm instance, x, and y data, train and evaluate and return results
def experiment(svm, x_train, y_train, x_test, y_test):
    
    # Fit on the training data, predict on the test
    svm.fit(x_train)
    
    preds = svm.predict(x_test)
    
    # Predictions are either -1 or 1
    preds = (preds < 0).astype(int)
    
    pre = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    return {'precision': pre, 'recall': rec, 'f1': f1}
    


# In[ ]:





# In[20]:


# Perform experimental search for best gamma parameter
validation_results = {}
gamma = [0.005, 0.05, 0.5]
for g in gamma:
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=0.0065, gamma=g)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[g] = res
    
                       


# In[21]:


# Printing out the results of the validation. 
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# In[22]:


# Perform experimental search for gamma with a narrower range. Looks like smaller gamma is better
validation_results = {}
# Search 1e-5, 5e-5, 1e-4, 1.5e-4, 1e-3, 1.5e-3 and 2e-3
gamma = [1, 5, 10, 15, 20, 100, 150, 200]
for g in gamma:
    g = g / 100000.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=0.0065, gamma=g)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[g] = res
    
                       


# In[23]:


# Printing out the results of the validation.
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# Let's try the gamma value of 1e-5 and now tune **nu**.

# In[24]:


# Perform experimental search for nu
validation_results = {}
nu = range(1, 10)
for n in nu:
    n = n / 1000.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=n, gamma=0.00001)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[n] = res
    
                       


# In[25]:


# Printing out the results of the validation. 
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# In[26]:


# Perform experimental search for nu with a finer range
validation_results = {}
nu = range(1, 11)#[0.005, 0.0065, 0.01, 0.015]
for n in nu:
    n = n / 10000.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=n, gamma=0.00001)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[n] = res
                           


# In[27]:


# Printing out the results of the validation. The optimal setting is between 512 and 2048
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# Now that we have found the optimal nu setting of 0.0002, let's try and train the model.
# 
# It turns out that gamma = 1e-5 was not performing that well once the data size increased. We upped it to 0.005, which produced favorable results and allowed the model to generalize better.

# In[28]:


# We increased gamma back to 0.005 as it helped with the fit once we increased the number of samples.
svm = OneClassSVM(nu=0.0002, gamma=0.005)
svm.fit(x_train_train[:])


# In[29]:


preds = svm.predict(x_test)
preds = (preds < 0).astype(int)


# In[30]:


pre = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Precision: {pre}")
print(f"Recall: {rec}")
print(f"F1-Measure: {f1}")


# In[31]:


ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot()


# Analyzing the confusion matrix, this model did very well. There are no false negatives, and only 17 false positives.
# 
# One-Class SVM is much more sensitive and responsive to dataset processing and hyperparameter tuning, but it is capable of producing strong results.

# In[ ]:




