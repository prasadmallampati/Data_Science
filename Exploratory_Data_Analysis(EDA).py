# -*- coding: utf-8 -*-
"""
# EDA ON CANCER DATASET
Created on Mon Jul 22 11:22:42 2024

@author: prasad


"""
'''
1. Understand the Data
Age: Age of the individual
Gender: Gender of the individual
BMI: Body Mass Index
Smoking: Smoking status (e.g., smoker/non-smoker)
GeneticRisk: Genetic risk factor (e.g., high/low)
PhysicalActivity: Level of physical activity (e.g., sedentary, active)
AlcoholIntake: Alcohol intake level (e.g., high/low)
CancerHistory: History of cancer in the family (e.g., yes/no)
Diagnosis: Whether cancer was diagnosed (target variable)

'''

# import req packages

import pandas as pd

# 2. Data Collection

 #Load data set
df=pd.read_csv('cencer.csv')

# Display the first few rows of the dataframe
df.head()

# 3 data cleaning

# means handling missing values

missing_val=df.isnull().sum()

missing_val
# if any fillna(dataset.median())
# if  any  dropna()

# remove duplicates

dup = df.duplicated().sum()

# Remove duplicate rows
df = df.drop_duplicates()
# priting duplicates 

print(dup)


# datatype

# Check data types for knowing which type it is  
print(df.dtypes)

# Converting existing datatype  to req
df['Gender'] = df['Gender'].astype('category')
df['Smoking'] = df['Smoking'].astype('category')
df['GeneticRisk'] = df['GeneticRisk'].astype('category')
df['PhysicalActivity'] = df['PhysicalActivity'].astype('category')
df['AlcoholIntake'] = df['AlcoholIntake'].astype('category')
df['CancerHistory'] = df['CancerHistory'].astype('category')
df['Diagnosis'] = df['Diagnosis'].astype('category')


df.dtypes

# 4. Data Exploration
# Descriptive Statistics

# descriptive sta for numarical
df.describe()

# descriptive stat for both 

df.describe(include=['category'])

# data distibution


import matplotlib.pyplot as plt
import seaborn as sns


# for numarical which is age and bmi


df[['Age','BMI']].hist(bins=30,figsize=(10,5))
plt.show()


plt.hist([df.Age,df.BMI], bins=30)
plt.show()


# Distribution of categorical features
sns.countplot(data=df, x='Gender')
plt.show()

sns.countplot(data=df, x='Smoking')
plt.show()

sns.countplot(data=df, x='Diagnosis')
plt.show()


# 5 visualization 
'''
1 Univariate
2.Bivariate 
3.Multivariate

'''
# 1 Univariate Analysis(individual var)

# Distribution of Age
sns.histplot(df['Age'], bins=30, kde=True)

# Boxplot for BMI
sns.boxplot(x='BMI', data=df)



# 2 Bivariate Analysis(relationship  between two variable)

# Age vs BMI
sns.scatterplot(x='Age', y='BMI', hue='Diagnosis', data=df)

# Gender vs Diagnosis
sns.countplot(data=df, x='Gender', hue='Diagnosis')


# 3 Multivariate Analysis( Analyze interactions between multiple variables)

# Pairplot to visualize relationships between variables
sns.pairplot(df, hue='Diagnosis')

# Correlation heatmap
corr = df[['Age', 'BMI']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

# 6 tests

from scipy import stats

# T-test for BMI between diagnosed and non-diagnosed
diagnosed_bmi = df[df['Diagnosis'] == 'Yes']['BMI']
non_diagnosed_bmi = df[df['Diagnosis'] == 'No']['BMI']

t_stat, p_value = stats.ttest_ind(diagnosed_bmi, non_diagnosed_bmi)

print(f"T-statistic: {t_stat}, P-value: {p_value}")

# 7 feature engineering
# Example: Creating an interaction feature between Age and BMI
df['Age_BMI_Interaction'] = df['Age'] * df['BMI']

# feature selection
correlation = df.corr()
important_features = correlation['Diagnosis'].abs().sort_values(ascending=False)
print(important_features)
