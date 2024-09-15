# Compute the Pearson Correlation between the features 'Age' and 'Duration of Stay' in Covid dataset
import pandas as pd
from scipy.stats import pearsonr

# Import your data into Python
df = pd.read_csv("Covid_data.csv")
 
# Convert dataframe into series
list1 = df['Age']
list2 = df['DaysOfStay']
 
# Apply the pearsonr()
corr, _ = pearsonr(list1, list2)
print('Pearson correlation: %.3f' % corr)

# Pearson correlation: 0.205 (Moderate Positive correlation)
# Interpretaton:
# As the age of the patient increases, days of stay in hospital also increases

# Draw a Plot of the relationship
# 'Age' on the X Axis and 'Days of Stay' on the Y axis
from matplotlib import pyplot
pyplot.scatter(list1, list2)
pyplot.show()
