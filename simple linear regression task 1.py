#!/usr/bin/env python
# coding: utf-8

# # TASK:1
# ## Prediction using supervised ML
# ### Author: CHANDRU T
# #### BATCH #GRIP JUNE22
# 

# ###  STEP:1 Importing  the libraries

# In[2]:



import numpy as np 
import pandas as pd


# ### Reading data from remote link

# In[3]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)


# In[4]:


data.head()


# In[5]:


data.describe()


# ### Visualization view for the input data

# ###  import visualization library and checking the distribution of the  dataset

# In[6]:


import seaborn as sns 
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(7,5)})
sns.distplot(data.Scores, bins=30)
plt.show()


# ## **Preparing the data**
# 
# ### The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[7]:


X=data['Hours'].values
y=data['Scores'].values
print(X)
print(y)


#  ## **Training the Algorithm**
# ### We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[8]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
X_train = np.reshape(X_train, (20, 1))
y_train = np.reshape(y_train, (20, 1))
X_test = np.reshape(X_test, (5, 1))
regressor.fit(X_train, y_train) 
y_pred=regressor.predict(X_test)


# ### Plotting for the test data

# In[9]:


line=line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.show()


# ### **Making Predictions** Now that we have trained our algorithm, it's time to make some predictions.
# 

# 
# ## You can also test with your own data

# In[12]:


hours = 9.25
hours = np.reshape(hours, (1, 1))
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ## **Evaluating the model** The final step is to evaluate the performance of algorithm.

# In[11]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('\n\n\n Mean Squared error:  ',metrics.mean_squared_error(y_test,y_pred))
print('\n R2 Score: ',metrics.r2_score(y_test,y_pred))
print('\n Root mean squared error: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# ## THANK YOU
