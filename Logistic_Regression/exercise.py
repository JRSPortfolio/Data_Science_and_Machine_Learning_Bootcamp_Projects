'''
Logistic Regression Project
In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement.
We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Read in the advertising.csv file and set it to a data frame called ad_data.
ad_data = pd.read_csv('Logistic_Regression/advertising.csv')

# Check the head of ad_data
print(ad_data.head())

# Use info and describe() on ad_data
print(ad_data.info())
print(ad_data.describe())

# Create a histogram of the Age
sns.histplot(ad_data['Age'], kde = True, color = 'mediumvioletred') 

# Create a jointplot showing Area Income versus Age.
sns.jointplot(ad_data, x = 'Age', y = 'Area Income', color = 'salmon')

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
sns.jointplot(ad_data, x = 'Age', y = 'Daily Time Spent on Site', kind = 'kde', palette = 'darkgreen')

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(ad_data, x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', color = 'turquoise')

# Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(ad_data, diag_kind = 'hist', hue = 'Clicked on Ad', palette = 'YlGn', height = 1.65)

# Split the data into training set and testing set using train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# Train and fit a logistic regression model on the training set.
lg_model = LogisticRegression(max_iter = 200)
lg_model.fit(X_train, y_train)

# Now predict values for the testing data.
predictions = lg_model.predict(X_test)

# Create a classification report for the model.
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

plt.show()