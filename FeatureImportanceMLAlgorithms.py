# URL Ref - https://machinelearningmastery.com/calculate-feature-importance-with-python/
# For the Covid Dataset, show the feature importance for:
# 1. Decision Tree - CART Feature Importance
# 2. Random Forest
# 3. Permutation Feature Importance - KNN 

# Demonstrate various Feature Selection Techniques on the "Covid" dataset
# URL for reference - https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/

import pandas as pd
import numpy as np
 
df = pd.read_csv("Covid_data.csv")
print(df)

from sklearn import preprocessing
import matplotlib.pyplot as plt

# Encode the Categorical data of Gender - using Dummy Column - Creates a new column called Gender_M
# Use Dummy Variables
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

print(df.columns)
print(df.columns[0])
# Columns - Age [0], Co_Morbid [1], Admit_date [2], Discharge_date [3], Remdesevir_given [4], 
# DaysofStay [5], DischargeType [6], Covid_Severity [7], 
# Covid_SeverityDescription [8], DischargeTypeCategorical [9], Gender_M [10]

X = df.iloc[:,[0,1,4,5,7,10]]   #Age, Co_Morbid, Remdesevir_given, DaysofStay, Covid_Severity, Gender
Y = df.iloc[:,[6]]  #DischargeType

# 1. Decision Tree
# Decision tree algorithms like classification and regression trees (CART) offer importance scores based on the reduction in the criterion used to select split points, like Gini or entropy.
# After being fit, the model provides a feature_importances_ property that can be accessed to retrieve the relative importance scores for each input feature.
from sklearn.tree import DecisionTreeClassifier
# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X,Y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Results
# F0 [Age] - [0.42345]
# F1[Comorbid] - [0.02695]
# F2[Remdesevir] - [0.08221]
# F3[DaysofStay] - [0.30261]
# F4[Covid_Severity] - [0.11269]
# F5[Gender] - [0.05207]
# **Indicates that Age, DaysofStay, Covid_Severity are likely candidates for splitting the decision tree.

# 2. Random Forest Feature Importance
# After being fit, the model provides a feature_importances_ property that can be accessed to retrieve the relative importance scores for each input feature.
from sklearn.ensemble import RandomForestRegressor
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X,Y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Results
# F0 [Age] - [0.3960]
# F1[Comorbid] - [0.0425]
# F2[Remdesevir] - [0.04553]
# F3[DaysofStay] - [0.33038]
# F4[Covid_Severity] - [0.1295]
# F5[Gender] - [0.0560]
# **Indicates that Age, DaysofStay, Covid_Severity are likely candidates for splitting the decision tree.
# Results are in-tune with the decision tree approach

# 3. Permutation Feature Importance - using KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(X,Y)
# perform permutation importance
results = permutation_importance(model, X, Y, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Results - KNN
# F0 [Age] - [0.08377]
# F1[Comorbid] - [-0.00065]
# F2[Remdesevir] - [0.00162]
# F3[DaysofStay] - [0.05925]
# F4[Covid_Severity] - [0.02655]
# F5[Gender] - [-0.00422]
# **Indicates that Age, DaysofStay, Covid_Severity are likely candidates for splitting the decision tree.
# Results are in-tune with the decision tree approach and Random Forest