# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:40:08 2024

@author: Armanis
"""



#%%
# Import required packages for this example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import dmba
from dmba import classificationSummary


# Load the data

file_path = r"C:\Users\Armanis\OneDrive\Desktop\Python CSV Files\accidentsFull.csv"
accidents = pd.read_csv(file_path)

accidents.head(12)

print(accidents.columns) # Use to see columns quicker
#%%
# create a binary variable called injury (output variable)
accidents['INJURY'] = np.where(accidents['MAX_SEV_IR']>0,'yes','no') # If MAX_SEV_IR > 0: Assigns 'yes' Otherwise: Assigns 'no'
print(accidents.INJURY)


# proportion of "yes" and "no" in the Injury variable
accidents['INJURY'].value_counts()/len(accidents)


# determine variables to use in model
outcome = 'INJURY'
predictors = ['HOUR_I_R', 'ALIGN_I', 'WRK_ZONE', 'WKDY_I_R', 'INT_HWY', 'LGTCON_I_R', 'PROFIL_I_R', 'SPD_LIM',
              'SUR_COND', 'TRAF_CON_R', 'TRAF_WAY', 'WEATHER_R'] # speed limit, road alignment, work zone indicator, etc

x = pd.get_dummies(accidents[predictors]) # Converts categorical predictors into dummy variables. 

#Example:
# Suppose ALIGN_I has 3 categories: Straight, Curve, and Hill.
# The one-hot encoding will create three new columns: ALIGN_I_Straight, ALIGN_I_Curve, ALIGN_I_Hill.
# If a row has ALIGN_I = Curve, the values in these columns will be:

# ALIGN_I_Straight = 0
# ALIGN_I_Curve = 1
# ALIGN_I_Hill = 0

y = accidents['INJURY'].astype('category')

# Converts the INJURY column into a categorical variable:( it was a string before look at line 30)
# This makes it easier to handle as a target variable in the Naive Bayes model.


classes = list(y.cat.categories)

# Extracts the category labels ('no', 'yes') from the y variable.
# Stores these labels in a list called classes, which will be used later 


# create data partitions for the model (train = 60%)
x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size = .6, random_state = 1)

# fit the model
accidentsNB = MultinomialNB (alpha = 0.01)
accidentsNB.fit(x_train,y_train)

# calculate probabiliites for training and validations sets
predProb_train = accidentsNB.predict_proba(x_train)
predProb_valid = accidentsNB.predict_proba(x_valid)

# determine class membership based on probabiliites calculated
y_train_pred = accidentsNB.predict(x_train)
y_valid_pred = accidentsNB.predict(x_valid)


# create confusion matrix for training set
classificationSummary(y_train,y_train_pred, class_names = classes)

# create confusion matrix for validation set
classificationSummary(y_valid,y_valid_pred, class_names = classes)