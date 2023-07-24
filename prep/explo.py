# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('training_data.csv')

# Display the first few rows of the dataframe
print(df.head())

# Get a summary of the dataframe
print(df.info())

# Generate descriptive statistics of the dataframe
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Let's visualize the distributions of numeric features
numeric_features = df.select_dtypes(include=[np.number])
for col in numeric_features.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=numeric_features, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Correlation matrix to see relationships between different numeric features
corr_matrix = numeric_features.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
