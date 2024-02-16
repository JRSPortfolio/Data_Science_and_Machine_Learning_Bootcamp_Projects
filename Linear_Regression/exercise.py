'''
Linear Regression Project
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read in the Ecommerce Customers csv file as a DataFrame called customers.
customers = pd.read_csv('Linear_Regression/Ecommerce Customers')

# Check the head of customers, and check out its info() and describe() methods.
print(customers.head())
print('---------------------------------------------------------')
print(customers.info())
print('---------------------------------------------------------')
print(customers.describe())
print('---------------------------------------------------------')

fig, axes = plt.subplots(1, 2, figsize = (18, 8))
plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.1)

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns.
sns.jointplot(customers, x = 'Time on Website', y = 'Yearly Amount Spent', color = 'thistle')

# Do the same but with the Time on App column instead.
sns.jointplot(customers, x = 'Time on App', y = 'Yearly Amount Spent', color = 'greenyellow')

# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(customers, x = 'Time on App', y = 'Yearly Amount Spent', cmap = 'Wistia', kind = 'hex')

# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.
sns.pairplot(customers, height = 2)

# Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. 
sns.lmplot(customers, x = 'Length of Membership', y = 'Yearly Amount Spent')

# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column.
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website',
               'Length of Membership']]
y = customers['Yearly Amount Spent']

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101*
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# Train/fit lm on the training data.
lm.fit(X_train, y_train)

#*Print out the coefficients of the model
cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
print(cdf.head())
print(lm.coef_)
print('---------------------------------------------------------')

# Use lm.predict() to predict off the X_test set of the data.
predictions = lm.predict(X_test)

# Create a scatterplot of the real test values versus the predicted values.
sns.scatterplot(x = y_test, y = predictions, ax = axes[0], color = 'peru')
axes[0].grid(True, color = 'darkslateblue')
axes[0].set_title('Test & Predictions')

# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
print(f'MAE : {metrics.mean_absolute_error(y_test, predictions)}')
print(f'MSE : {metrics.mean_squared_error(y_test, predictions)}')
print(f'RMSE : {np.sqrt(metrics.mean_squared_error(y_test, predictions))}')
print(metrics.explained_variance_score(y_test, predictions))

# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.histplot((y_test - predictions), kde = True, bins = 50, ax = axes[1], color = 'darkgoldenrod')
axes[1].grid(True, color = 'maroon')
axes[1].set_title('Residuals')

plt.show()