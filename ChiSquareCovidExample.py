# Code to demonstrate ChiSquare for the Covid Dataset - Categorical data
import pandas as pd
import numpy as np
 
df = pd.read_csv("Covid_data.csv")
print(df)

# To explore the correlation between Gender and Covid_SeverityDescription in the dataset 
new = df.groupby(['Covid_SeverityDescription','Gender']).size()
print(new)
# mild - 101 [female], 149 [male] 
# moderate - 72 [female],124 [male]
# severe - 145 [female], 251 [male]
# undetermined - 175 [female], 215 [male]

from scipy.stats import chi2_contingency
data = [101,72,145,175], [149,124,251,215]
stat, p, dof, expected = chi2_contingency(data)

## interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (H0 holds true)')

# Output: p value is 0.0845; Independent (H0 holds True)
# alpha of 0.05 < 0.0845

