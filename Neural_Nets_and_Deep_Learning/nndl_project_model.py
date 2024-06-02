'''
Keras API Project Exercise

Goal

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off),
can we build a model thatcan predict wether or nor a borrower will pay back their loan?
This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.

The "loan_status" column contains our label.



Model creation section
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
import random

fig01, axes01 = plt.subplots(3, 1, figsize = (18, 9))

#get previous data
df = pd.read_csv('Neural_Nets_and_Deep_Learning/lending_club_loan_engineered.csv', index_col = 0)
print(df.head())
print(df.info())
print('____________________\n____________________')

# Set X and y variables to the .values of the features and label.
X = df.drop('loan_repaid', axis = 1)
y = df['loan_repaid']

# Perform a train/test split with test_size=0.2 and a random_state of 101.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# Use a MinMaxScaler to normalize the feature data X_train and X_test.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Build a sequential model to will be trained on the data.

model = Sequential()
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['precision', 'recall'])

# early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)


# Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting.
model.fit(x = X_train, y = y_train, epochs = 4, validation_data = (X_test, y_test), batch_size = 128)

# Plot out the validation loss versus the training loss.
losses = pd.DataFrame(model.history.history)
sns.lineplot(losses, ax = axes01[0])
axes01[0].set_xticks(range(0, 4))
axes01[0].set_yticks([i / 10 for i in range(0, 11)])
axes01[0].grid(True)

# Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype('int32')

print(f'Classification Report:\n{classification_report(y_test, predicted_classes)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, predicted_classes)}')
print('____________________\n____________________')

model = Sequential()
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'precision', 'recall'])

# early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)


# Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting.
model.fit(x = X_train, y = y_train, epochs = 3, validation_data = (X_test, y_test), batch_size = 128)

# Plot out the validation loss versus the training loss.
losses = pd.DataFrame(model.history.history)
sns.lineplot(losses, ax = axes01[1])
axes01[1].set_xticks(range(0, 3))
axes01[1].set_yticks([i / 10 for i in range(0, 11)])
axes01[1].grid(True)

# Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype('int32')

print(f'Classification Report:\n{classification_report(y_test, predicted_classes)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, predicted_classes)}')
print('____________________\n____________________')


model = Sequential()
model.add(Dense(160, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'precision', 'recall'])

# early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)


# Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting.
model.fit(x = X_train, y = y_train, epochs = 2, validation_data = (X_test, y_test), batch_size = 128)

# Plot out the validation loss versus the training loss.
losses = pd.DataFrame(model.history.history)
sns.lineplot(losses, ax = axes01[2])
axes01[2].set_xticks(range(0, 2))
axes01[2].set_yticks([i / 10 for i in range(0, 11)])
axes01[2].grid(True)

# Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype('int32')

print(f'Classification Report:\n{classification_report(y_test, predicted_classes)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, predicted_classes)}')
print('____________________\n____________________')


# Given the customer below, would you offer this person a loan?
random.seed(101)
random_ind = random.randint(0,len(df))
new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer

random.seed(101)
random_ind = random.randint(0,len(df))
new_customer = df.drop('loan_repaid', axis = 1).iloc[random_ind]
new_customer = new_customer.to_frame().transpose()
print(new_customer)
print(type(new_customer))
print('____________________\n____________________')
new_customer = scaler.fit_transform(new_customer)
nc_pred = model.predict(new_customer)
nc_pred_class = (nc_pred > 0.5).astype('int32')
if nc_pred_class == 1:
    print(f'Prediction Result: {nc_pred_class} - Loan Approved')
else:
    print(f'Prediction Result: {nc_pred_class} - Loan Rejected')
print('____________________\n____________________')

# Now check, did this person actually end up paying back their loan?
print(df['loan_repaid'].iloc[random_ind])
print('____________________\n____________________')

plt.show()