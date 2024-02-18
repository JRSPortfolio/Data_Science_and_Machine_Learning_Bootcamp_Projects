'''
Support Vector Machines Project 

The Data
For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') 
iris = sns.load_dataset('iris')
print(iris.head(), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')

# Create a pairplot of the data set.
sns.pairplot(iris, diag_kind = 'hist', hue = 'species', height = 2.2, palette = 'bright')

# Create a kde plot of sepal_length versus sepal width for setosa species of flower.
plt.figure(figsize = (15, 8))
sns.kdeplot(iris[iris['species'] == 'setosa'], x = 'sepal_width', y = 'sepal_length', cmap = 'plasma', fill = True, thresh = 0.1)

# Split your data into a training set and a testing set.
X = iris.drop('species', axis = 1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# all the SVC() model from sklearn and fit the model to the training data.
svc_mod = SVC()
svc_mod.fit(X_train, y_train)

# Now get predictions from the model and create a confusion matrix and a classification report.
predictions = svc_mod.predict(X_test)
print(confusion_matrix(y_test, predictions), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(classification_report(y_test, predictions), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')

# Create a dictionary called param_grid and fill out some parameters for C and gamma
param_grid = {'C' : [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'gamma' : [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]}

# Create a GridSearchCV object and fit it to the training data.
grid = GridSearchCV(SVC(), param_grid, verbose = 2)
grid.fit(X_train, y_train)

# Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them.
print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(classification_report(y_test, grid_predictions))

plt.show()