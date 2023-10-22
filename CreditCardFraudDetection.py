#!/usr/bin/env python
# coding: utf-8

# # Importing Important Libraries 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing DataSet 

# In[2]:


data= pd.read_csv('creditcard.csv')


# # Head of DataSet

# In[3]:


data.head(10)


# # Shape and Describtion of Dataset

# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.keys()


# # To Know lf The DataSet Has Missing Values

# In[7]:


missing_values = data.isnull().sum()
print(missing_values)


# # Histograms 

# In[8]:


for column in data.columns:
    if data[column].dtype in [int, float]:
        plt.figure(figsize=(8, 6))
        plt.hist(data[column], bins=20)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


# # ScaterPlots
# 

# In[9]:


sns.scatterplot(data=data, x='Time', y='V1', hue='Class')


# In[10]:


sns.scatterplot(data=data, x='V2', y='V3', hue='Class')


# In[11]:


sns.scatterplot(data=data, x='V4', y='V5', hue='Class')


# In[12]:


sns.scatterplot(data=data, x='V6', y='V7', hue='Class')


# In[13]:


sns.scatterplot(data=data, x='V8', y='V9', hue='Class')


# In[14]:


sns.scatterplot(data=data, x='V10', y='V11', hue='Class')


# In[15]:


sns.scatterplot(data=data, x='V12', y='V13', hue='Class')


# In[16]:


sns.scatterplot(data=data, x='V14', y='V15', hue='Class')


# In[17]:


sns.scatterplot(data=data, x='V16', y='V17', hue='Class')


# In[18]:


sns.scatterplot(data=data, x='V18', y='V19', hue='Class')


# In[19]:


sns.scatterplot(data=data, x='V20', y='V21', hue='Class')


# In[20]:


sns.scatterplot(data=data, x='V22', y='V23', hue='Class')


# In[21]:


sns.scatterplot(data=data, x='V24', y='V25', hue='Class')


# In[22]:


sns.scatterplot(data=data, x='V26', y='V27', hue='Class')


# In[23]:


sns.scatterplot(data=data, x='V28', y='Amount', hue='Class')


# In[24]:


X=data.drop('Class', axis=1)
y=pd.Series(data.Class)


# In[25]:


plt.figure(figsize=(12,12))
corre=X.corr()
sns.heatmap(corre, annot=True, cmap='coolwarm')


# In[26]:


print(corre)


# # Spliting Data Into Train/Test

# In[27]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# # Preprocessing & Dimentionality Reduction

# In[28]:


from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 


# #  Model Support Vector Machine Training 

# In[29]:


from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC
n_components=10


# In[30]:


pipeline=Pipeline([
    ('scaler',StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('classifiers', SVC())
])


# In[ ]:


pipeline.fit(X_train, y_train)


# # Testing Model

# In[ ]:


y_pred=pipeline.predict(X_test)


# In[ ]:


accuracy= pipeline.score(X_test, y_test)


# In[ ]:


print(accuracy)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print(confusion)


# # Tuning Hyperparameters 

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


para_grid= {
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1.0]
}
grid_search= GridSearchCV(pipeline,para_grid,cv=5,scoring='accuracy')


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:




