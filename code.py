# --------------
# Code starts here
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import Imputer, LabelEncoder
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#### Data 1
# Load the data
df = pd.read_csv(path)
#print(df.head())
# Overview of the data
#print(df.head())
#print(df.describe())

# Histogram showing distribution of car prices
sns.distplot(df['price'], kde=True)

# Countplot of the make column
plt.figure(figsize=(12,6))
sns.countplot(y = 'make', data=df)
# Jointplot showing relationship between 'horsepower' and 'price' of the car
sns.jointplot('horsepower', 'price', data=df, kind='reg')

# Correlation heat map
plt.figure(figsize=(14,7))
sns.heatmap(df.corr(), cmap='viridis')

# boxplot that shows the variability of each 'body-style' with respect to the 'price'
plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
sns.boxplot('body-style', 'price', data=df)

#### Data 2

# Load the data
df1 = pd.read_csv(path2)
print(df1.info())
print(df1.describe())

# Impute missing values with mean
df1 = df1.replace('?', 'NaN')
#print(df1.isnull().sum())
numeric_imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df1['normalized-losses'] = numeric_imp.fit_transform(df1[['normalized-losses']])
df1['horsepower'] = numeric_imp.fit_transform(df1[['horsepower']])
print('horsepower')
#print(df1.columns)

# Skewness of numeric features
numeric_features_auto = df1._get_numeric_data().columns
print(df1['horsepower'].head())


# Label encode 
def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding' + feature)
    return df    

print(df1['make'].unique())
df1 = dummyEncode(df1)
print(df1['make'].unique())

#Combine teh height & width
df1['area'] = df1['height']*df1['width']

# Code ends here


