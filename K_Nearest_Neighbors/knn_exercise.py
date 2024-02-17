'''
K Nearest Neighbors Project
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Read the 'KNN_Project_Data csv file into a dataframe
df = pd.read_csv('K_Nearest_Neighbors/KNN_Project_Data')

# Check the head of the dataframe.
print(df.head())

# Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.
sns.pairplot(df, diag_kind = 'hist', hue = 'TARGET CLASS', palette = 'YlGnBu', height = 1.2)

# Create a StandardScaler() object called scaler.
scaler = StandardScaler()

# Fit scaler to the features.
scaler.fit(df.drop('TARGET CLASS', axis = 1))

# Use the .transform() method to transform the features to a scaled version.
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))

# Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.
feat_df = pd.DataFrame(scaled_features, columns = df.columns[:-1])
print(feat_df.head())

# Use train_test_split to split your data into a training set and a testing set.
X = feat_df
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# Create a KNN model instance with n_neighbors=1
knn = KNeighborsClassifier(n_neighbors = 1)

# Fit this KNN model to the training data.
knn.fit(X_train, y_train)

# Use the predict method to predict values using your KNN model and X_test.
predictions = knn.predict(X_test)

# Create a confusion matrix and classification report.
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list.
error_rate = []
for i in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Now create the following plot using the information from your for loop.
plt.figure(figsize = (16, 8))
plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.1)
sns.lineplot(x = range(1, 101), y = error_rate, color = 'darkmagenta', linestyle = '-', marker = '*', markerfacecolor = 'olivedrab', markersize = 12)
plt.xticks(range(0, 101, 2))
plt.yticks([i / 100 for i in range(15, 31)])
plt.grid(True)
plt.title('Error Rate - Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.
best_k_values = [31, 37, 39, 53, 54]
for k in best_k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    print(k)
    print(confusion_matrix(y_test, pred_k))
    print(classification_report(y_test, pred_k))
    print('---------------------------------------------------')

plt.show()