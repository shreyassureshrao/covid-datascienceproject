# Sample Univariate Visualization in Python - Single Column

from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# set default theme
sns.set_theme()

# Import your data into Python
df = pd.read_csv("Covid_data.csv")
print(df.index)

# --------------------------------------- UNIVARIATE ANALYSIS ------------------------------

# 1.1 Box Plot
sns.boxplot(df['Age'])   # alternative is plt.boxplot(df['Age'])
plt.title('1. Box Plot of Age')
plt.show()

#1.2 strip plot is used to visualize the distribution of data points of a single variable
sns.stripplot(y=df['Age'])
plt.title('2. Strip Plot of Age')
plt.show()

#1.3 Swarm Plot - a visualization technique for univariate data to view the spread of values 
# in a continuous variable.
# For Age
sns.swarmplot(x=df['Age'])
plt.title('3. Swarm Plot of Age')
plt.show()

# For Days of Stay
sns.swarmplot(x=df['DaysOfStay'])
plt.title('4. Swarm Plot of Days of Stay')
plt.show()

#1.4 Histograms
plt.hist(df['Age'])
plt.title('5. Histogram of Age')
plt.show()

# 1.5 SNS distplot to plot a histogram
sns.distplot(df['Age'],kde=FALSE, color='blue',bins=5)
plt.title('6. Dist Plot of Age with 5 bins')
plt.show()

# 1.6 countplot - visualizing categorical variables
sns.countplot(df['Gender'])
plt.title('7. Count Plot of Gender (Categorical)')
plt.show()

# --------------------------------------- BIVARIATE ANALYSIS -----------------------------

#2.1 Boxplot - visualize the min, max, median, IQR, outliers of a variable
# Covid Severity 1-> Mild; 2-> Moderate; 3->Severe; 4-> Undetermined
sns.boxplot(x=df['Covid_Severity'],y=df['DaysOfStay'],data=df) 
plt.title('8. Box Plot of Covid Severity vs Days of Stay')
plt.show()

#2.2 Scatter Plot
# Visualize the relationship between two variables
sns.scatterplot(x=df['DaysOfStay'],y=df['Age'])
plt.title('9. Scatter Plot of Age vs Days of Stay')
plt.show()

# Hue will indicate which field will have the color coding
sns.scatterplot(x=df['DaysOfStay'],y=df['Age'],hue=df['Covid_SeverityDescription'])
plt.title('10. Scatter Plot of Age vs Days of Stay vs Covid Severity Description (hue value)')
plt.show()

#2.3 FacetGrid 
# Gender vs Discharge Type distribution plot
g = sns.FacetGrid(df, col="Gender", height=6.5, aspect=.85)
g.map(sns.histplot, "DischargeType")
plt.title('11. Facet Grid of Gender vs Discharge Type')
plt.show()

#----------------------------- MULTIVARIATE ANALYSIS ----------------------------------

# DischargeTypeCategorical vs Age vs Gender distribution plot
g = sns.FacetGrid(df, col="DischargeTypeCategorical", hue="Gender", margin_titles=True, height=6.5, aspect=.85)
g.map(sns.histplot, "Age")
plt.title('12. Facet Grid of Gender vs Age vs Discharge Type')
plt.show()

# 2.4 lmplot
# Age vs Gender vs DischargeType
sns.lmplot(data=df, x="Age", y="DischargeType",hue="Gender")
plt.title('13. lmplot of Age vs Discharge Type vs Gender (hue)')
plt.show()

# Days of Stay vs Gender vs Age
sns.lmplot(data=df, x="Age", y="DaysOfStay",hue="Gender")
plt.title('14. lmplot of Age vs Days of Stay vs Gender (hue)')
#plt.show()

# Days of Stay vs DischargeTypeCategorical vs Age
sns.lmplot(data=df, x="DaysOfStay", y="Age",hue="DischargeTypeCategorical")
plt.title('15. lmplot of Age vs Days of Stay vs Discharge Type (hue)')
#plt.show()




