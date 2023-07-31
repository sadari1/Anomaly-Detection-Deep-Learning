#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import seaborn as sns
print('seaborn: ', sns.__version__)


# In[2]:


columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", 
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

df = pd.read_csv("data/kddcup.data.corrected", sep=",", names=columns, index_col=None)


# In[3]:


# Filter to only 'http' attacks
df = df[df["service"] == "http"]
df = df.drop("service", axis=1)


# In[4]:


df['label'] = df['label'].apply(lambda x: 0 if x=='normal.' else 1)
df['label'].value_counts()


# In[5]:


datatypes = dict(zip(df.dtypes.index, df.dtypes))

encoder_map = {}
for col, datatype in datatypes.items():
    if datatype == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoder_map[col] = encoder 


# In[6]:


# Check the variables with highest correlation with 'label'
df2 = df.copy()
label_corr = df2.corr()['label']


# In[7]:


# Filter out anything that has null entry or is not weakly correlated
train_cols = label_corr[(~label_corr.isna()) & (np.abs(label_corr) > 0.2)]
train_cols = list(train_cols[:-1].index)
train_cols


# In[8]:


labels = df2['label']
# Conduct a train-test split    
x_train, x_test, y_train, y_test = train_test_split(df2[train_cols].values, labels.values, test_size = 0.15, random_state = 42)


# In[9]:


# Additional split of training dataset to create validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[10]:


print("Shapes")
print(f"x_train:{x_train.shape}\ny_train:{y_train.shape}")
print(f"\nx_val:{x_val.shape}\ny_val:{y_val.shape}")
print(f"\nx_test:{x_test.shape}\ny_test:{y_test.shape}")


# In[11]:


import tensorflow
tensorflow.__version__


# In[12]:


from tensorflow.keras.utils import to_categorical

y_train =  to_categorical(y_train)
y_test =  to_categorical(y_test)
y_val =  to_categorical(y_val)


# In[13]:


print("Shapes")
print(f"x_train:{x_train.shape}\ny_train:{y_train.shape}")
print(f"\nx_val:{x_val.shape}\ny_val:{y_val.shape}")
print(f"\nx_test:{x_test.shape}\ny_test:{y_test.shape}")


# In[14]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping


# In[25]:


# The input layer requires you to specify the dimensionality of the x-features (and not the number of samples)
input_layer = Input(shape=(13))
h1 = Dense(26, activation='relu', kernel_initializer = 'he_uniform', kernel_regularizer = L2(l2=1e-5))(input_layer)
h2 = Dense(26, activation='relu', kernel_initializer = 'he_uniform', kernel_regularizer = L2(l2=1e-5))(h1)
h3 = Dense(26, activation='relu', kernel_initializer = 'he_uniform', kernel_regularizer = L2(l2=1e-5))(h2)
h4 = Dense(6, activation='relu', kernel_initializer = 'he_uniform', kernel_regularizer = L2(l2=1e-5))(h3)
output_layer = Dense(2, activation='softmax', kernel_regularizer = L2(l2=1e-5))(h4)

# Creating a model by specifying the input layer and output layer
model = Model(input_layer, output_layer)


# In[26]:


es = EarlyStopping(patience=5, min_delta=1e-3, monitor='val_loss', restore_best_weights=True)

callbacks = [es]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')


# In[27]:


epochs = 20
batch_size = 128

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)


# In[28]:


model.evaluate(x_test, y_test)


# In[35]:


preds = model.predict(x_test)

# One hot to the original label encodings
y_true = y_test.argmax(axis=1)

# Derive the label predictions from the probability scores
y_preds = preds.argmax(axis=1)

# Compute precision, recall, f1 scores
precision = precision_score(y_true, y_preds)
recall = recall_score(y_true, y_preds)
f1_measure = f1_score(y_true, y_preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Measure: {f1_measure}")


# In[33]:


roc_auc_score(y_true, y_preds)


# In[34]:


cm = confusion_matrix(y_true, y_preds)
plt.title("Confusion Matrix")
ax = sns.heatmap(cm, annot=True, fmt='0.0f')
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')


# In[ ]:




