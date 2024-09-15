# Demo of Normalization -> Min-Max and Z-Score Normalization 

# 1. Apply Min-Max Normalization to 'Age' column in the Covid dataset; 
# So age ranges will be in the interval [0,1]

import pandas as pd

df = pd.read_csv("Covid_data.csv")

df.head()

# copy the data
df_min_max_scaled = df.copy()

# apply normalization techniques
#for column in df_min_max_scaled.columns:
# new-x = x - min(x) / max(x) - min(x)
df_min_max_scaled['Age'] = (df_min_max_scaled['Age'] - df_min_max_scaled['Age'].min()) / (df_min_max_scaled['Age'].max() - df_min_max_scaled['Age'].min())	

# view normalized data
lst = []
for val in df_min_max_scaled['Age']:
  lst.append(val) 

formatted_lst = ['%.2f' % elem for elem in lst]

#print(df_min_max_scaled['Age'])
print(formatted_lst)

""" # 2. Z-Score Normalization
from scipy.stats import zscore

# Calculate the zscores and drop zscores into new column
# So age ranges will be in the interval [-1,+1]
df['Age_zscore'] = zscore(df['Age'])
#print(df['Age_zscore'])

# view normalized data
lst = []
for val in df['Age_zscore']:
  lst.append(val) 

formatted_lst = ['%.2f' % elem for elem in lst]

print(formatted_lst) """