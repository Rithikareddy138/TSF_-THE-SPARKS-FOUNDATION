#!/usr/bin/env python
# coding: utf-8

# # GRIP @ The sparks Foundation

# # NAME : GADE MARY RITHIKA REDDY

# # Task 1:

# # objective : To predict the score of student based on the number of study hours

# # Steps Involved:

# # 1.Data Acquisition

# # 1.1 importing necessary packages

# In[5]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as stat_model
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')


# ## 1.2 Reading the data from dataset

# In[6]:



data=pd.read_csv("http://bit.ly/w-data")


# # 1.3 Retrieving the dimensionality of the data frame

# In[7]:


print(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns")


# # 1.4 To display the first 5 and last 5 records of the dataset

# In[8]:


data.head()


# In[9]:



data.tail()


# # 2. Data preprosessing & Preparation

# # 2.1 To retrieve general characteristics of the dataset

# In[11]:


data.info()


# # 2.2  To retrieve statistical characteristics of the dataset

# In[12]:


data.describe()


# # 2.3 Checking for unique data values

# In[13]:



print(data['Hours'].unique())


# In[14]:


print(data['Scores'].unique())


# # 2.4 Checking for missing values in the dataset

# In[15]:


data.isnull().sum()


# # 2.5 Checking for duplicate values in the dataset

# In[16]:


data.duplicated().sum()


# # 3. Exploratory Data Analysis

# In[18]:


sns.pairplot(data)


# In[19]:


sns.regplot(x=data['Hours'],y=data['Scores'])


# In[20]:


sns.heatmap(data.corr(),annot=True)


# In[21]:



sns.boxplot(data['Hours'])


# In[22]:



sns.boxplot(data['Scores'])


# # 4. Model building

# In[23]:


x = data.iloc[:,0:-1].values
y = data.iloc[:,1].values


# # 4.1 splitting the dataset into train set and test set

# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# # 4.2 Training the data using Linear Regression

# In[25]:


regressor=stat_model.OLS(y_train,x_train).fit()


# # 4.3 Comparing predicted values with respect to the actual values

# In[26]:


prediction=regressor.predict(x_test)


# In[27]:


predict_show=pd.DataFrame({'Actual Score':y_test,'Predicted Score':prediction})
print(predict_show)


# # 4.4 Summary of the model

# In[28]:


regressor.summary2()


#  # 5 Predicting the score for 9.25 hours of study using this model

# In[29]:


Hours=9.25
Score=regressor.predict(Hours)
int_score=int(Score)
print(f"The predicted score is {int_score} if the student studies for {Hours} a day")

