# Demonstrate One-hot encoding and Label encoding in Python
# 1. Importing the Libraries
import pandas as pd
import numpy as np
 
# 2. Reading the file
df = pd.read_csv("Covid_data.csv")
print(df)

#3. Apply Label encoding to the field 'Gender'
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()  
df['Gender']=le.fit_transform(df['Gender'])
print(df['Gender'])

# For Covid_Severity field
le = LabelEncoder()  
df['Covid_Severity']=le.fit_transform(df['Covid_SeverityDescription'])
print(df['Covid_Severity'])

# Represent Gender using One-Hot Encoding
# importing one hot encoder 
print(df['Gender'].value_counts())
from sklearn.preprocessing import OneHotEncoder
one_hot_encoded_data = pd.get_dummies(df, columns = ['Gender'])
print(one_hot_encoded_data)

# One hot encoding for Covid_Severity field
print(df['Covid_Severity'].value_counts())
from sklearn.preprocessing import OneHotEncoder
one_hot_encoded_data = pd.get_dummies(df, columns = ['Covid_Severity'])
print(one_hot_encoded_data)

# For multiple columns
#one_hot_encoded_data = pd.get_dummies(df, columns = ['Covid_Severity', 'Gender'])
#print(one_hot_encoded_data)