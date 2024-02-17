#!/usr/bin/env python
# coding: utf-8

# # AIAP Project Score MLP for Deployment

# ## We pick the best model to be RF Regressor and we will define our pipeline for deployment using make_pipeline in six stages:
# 
# (1) Get user input for url of train data, use SQL Lite to retrieve data and pass to dataframe 'df'
# 
# (2) Create a ColumnTransformer that will transform the data into numerical variables and categorical variables, including ONE-Hot Encoder where required.
# 'number_of_siblings' - num 
# 'direct_admission' - cat (boolean: Yes, No)
# 'learning_style' - cat (boolean: Visual, Auditory)
# 'tuition' - cat (boolean: Yes, No)
# 'n_male' - num
# 'n_female' - num
# 'hours_per_week' - num
# 'attendance_rate' - num
# 'CCA_O' - cat (ONE-Hot encoder, drop CCA_A, CCA_S, CCA_C)
# 
# (3)Perform data preprocessing: 
# - Drop the columns that are not required for model training, leaving 9 features and y variable
# - Remove NaN values or Impute values where appropriate
# 
# (4) Scale the data using StandardScaler()
# 
# (5) Pass through the RF Regressor to train the model 
# 
# (6) Get user input for url of data for prediction. Run the model and get the predicted value for final_test scores. Export final_test scores to csv file on local file_path

# In[1]:


# basic


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import ast as ast

import math
from enum import Enum
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm

# mute warnings
import warnings
warnings.filterwarnings("ignore")

# scikit-learn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

# import category encoders
get_ipython().system('pip install category_encoders')
import category_encoders as ce

# ML Algo
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor

# SQL Lite
import sqlite3
import requests
import sqlalchemy



# In[3]:


# Ask User for URL to data file and fetch data from URL
# https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db

def get_url_from_user():
    """
    Function to ask the user to input a URL.
    
    Returns:
    url (str): The URL inputted by the user.
    """
    url = input("Please enter the URL to your Training Data: ")
    return url

# Example usage:
if __name__ == "__main__":
    user_url = get_url_from_user()
    print("You entered:", user_url)
    
    response = requests.get(user_url)
    score_data = sqlite3.connect('score.db')
    ## table name is known
    db_name = "score.db"
    table_name = "score"  ## insert table name
    engine = sqlalchemy.create_engine('sqlite:///%s'% db_name, execution_options={"sqlite_raw_colnames":True})
    print("Table Name:", table_name)
    df = pd.DataFrame()
    df = pd.read_sql_table(table_name, engine)
    df = df.dropna()
    df = df.drop_duplicates(subset='student_id')
    df.columns = [str(col) for col in df.columns]

    print(df)


# In[5]:


# There are now 19 columns , including dummies. Move column 'final_test' to the end
df = df[[col for col in df if col != 'final_test'] + ['final_test']]

# In[6]:


## Apply Train-test split to data
from sklearn.model_selection import train_test_split
X = pd.DataFrame()
X = df.iloc[ : , :-1]
y = pd.DataFrame()
y = df.iloc[ : , -1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[9]:


## import BaseEstimator. Grid Search and Pipelines: BaseEstimator is crucial for functionalities like grid search (GridSearchCV) and pipelines (Pipeline). Grid search uses get_params() to explore a range of hyperparameters, while pipelines rely on fit() and predict() to chain together multiple estimators.

from sklearn.base import BaseEstimator

## create class CCAEncoder to apply ONE_Hot Encoder to CCA = None

class CCAEncoder(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, df):
        df['CCA_O'] = (df['CCA'] == 'None')*1
        df['learning_style_visual'] = (df['learning_style'] == 'Visual')*1
        df['tuition_Yes'] = (df['tuition'] == 'Yes')*1
        df['direct'] = (df['direct_admission'] == 'Yes')*1

        return df


# In[10]:


## Where necessary, impute NaN values with strategy='mean' or 'most frequent' or others
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')



# In[11]:


# Define pre_process using ColumnTransformer to fit and transform

pre_process = ColumnTransformer(remainder='passthrough',
                                transformers=[('drop_columns', 'drop', ['sleep_time',
                                                                        'wake_time',
                                                                        'gender',
                                                                        'student_id',
                                                                        'age',
                                                                        'tuition',
                                                                        'direct_admission',
                                                                        'learning_style',
                                                                        'mode_of_transport',
                                                                        'CCA',
                                                                        'bag_color'])])
                                           


# In[12]:


model_pipe = make_pipeline(
    CCAEncoder(), # One_Hot Encoder for CCA = None
    pre_process, # apply Column Transformer
    StandardScaler(), # apply StandardScalar
    RandomForestRegressor(max_depth=10)# specify the training model
)


# In[13]:


model_pipe


# In[14]:


# Train the pipeline
model_pipe.fit(X_train, y_train)


# In[15]:

## Evaluate the model_pipe performance using R-Squared and RMSE

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Test the pipeline
y_pred = model_pipe.predict(X_test)

# Evaluate performance of the test set
r2_score_test = r2_score(y_test, y_pred)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred)))

print("Model_pipe r2_score_test: ", r2_score_test)
print("Model_pipe rmse_test: ", rmse_test)


# In[16]:


# Ask User for URL to prediction data file and fetch prediction data from URL
# https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db

def get_url_from_user():
    """
    Function to ask the user to input a URL.
    
    Returns:
    url (str): The URL inputted by the user.
    """
    url = input("Please enter the URL to your Prediction Data: ")
    return url

# Example usage:
if __name__ == "__main__":
    user_url = get_url_from_user()
    print("You entered:", user_url)

    response = requests.get(user_url)
    score_data = sqlite3.connect('score.db')
    ## table name is known
    db_name = "score.db"
    table_name = "score"  ## insert table name
    engine = sqlalchemy.create_engine('sqlite:///%s'% db_name, execution_options={"sqlite_raw_colnames":True})
    print("Table Name:", table_name)
    df1 = pd.DataFrame()
    df1 = pd.read_sql_table(table_name, engine)
    df1 = df.dropna()
    df1 = df.drop_duplicates(subset='student_id')
    df1.columns = [str(col) for col in df.columns]

    print(df1)


# In[17]:


# There are now 19 columns , including dummies. 'final_test' column should be blank, so move column 'final_test' to the end
df1 = df1[[col for col in df if col != 'final_test'] + ['final_test']]

## Drop final_test from Prediction Data 
X = pd.DataFrame()
X = df1.iloc[ : , :-1]
X.set_index('student_id', inplace=True)  ## preserve student_id so we can match our predicted final_test score to a specific student

X

# In[18]:


predict_pipe = make_pipeline(
    CCAEncoder(), # One_Hot Encoder for CCA = None
    pre_process, # apply Column Transformer
    StandardScaler(), # apply StandarScaler
    model_pipe.predict(X), # pass Prediction Data through trained model
    print(model_pipe.predict(X)) # get predictions
    
)


# In[19]:


y_prediction = model_pipe.predict(X)
y_p = pd.DataFrame()
y_p = pd.DataFrame(y_prediction)
X_index = pd.DataFrame(X.index)  ## put the student_id back to the predicted final_test scores
y_df = pd.concat([X_index, y_p], axis=1)


# In[22]:


# Ask User for file_path for saving y_prediction df
# C:\Users\hclee\Desktop\prediction.csv

def get_url_from_user():
    """
    Function to ask the user to input a file_path.
    
    Returns:
    url (str): The URL inputted by the user.
    """
    url = input("Please enter the file_path with final name ending in csv on your local drive where you want to save your final_test score prediction")
    
    return url

# Example usage:
if __name__ == "__main__":
    file_path = get_url_from_user()
    print("You entered:", file_path)


# In[23]:


y_df.to_csv(file_path, index=False)

# retrieve y_df to make sure it was saved
y_df_final = pd.read_csv(file_path)
y_df_final


# In[24]:


import joblib  ## save our model_pipe for future use
joblib.dump(model_pipe, "model_pipe.joblib")


# In[25]:

# retrieve model_pipe to make sure it is properly saved 

model_pipe2 = joblib.load("model_pipe.joblib")
model_pipe2


# In[26]:


get_ipython().run_line_magic('whos', '')


# In[ ]:




