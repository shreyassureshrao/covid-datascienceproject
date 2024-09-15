import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Covid_data.csv")
print(df)

# Convert date columns to datetime format
df['Admit_date'] = pd.to_datetime(df['Admit_date'], errors='coerce', dayfirst=True)
df['Discharge_date'] = pd.to_datetime(df['Discharge_date'], errors='coerce', dayfirst=True)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data type information
print("\nData Types:")
print(df.dtypes)