'''
Keras API Project Exercise

Goal

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off),
can we build a model thatcan predict wether or nor a borrower will pay back their loan?
This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.

The "loan_status" column contains our label.



Missing Values Section
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.models import load_model #type: ignore


#Recreating Dataframe format
df = pd.read_csv('Neural_Nets_and_Deep_Learning/lending_club_loan_two.csv')
df['loan_repaid'] = df['loan_status'].apply(lambda i: 1 if i == 'Fully Paid' else 0)
df.drop('emp_title', axis = 1, inplace = True)
df.drop('emp_length', axis = 1, inplace = True)

print(df)
print('____________________\n____________________')
print(f'{df.shape}\n{df.info()}')


fig01, axes01 = plt.subplots(2, 2, figsize = (18, 9))
plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.96, bottom = 0.06)


# Revisit the DataFrame to see what feature columns still have missing data.
print(df.isna().sum())
print('____________________\n____________________')

# Review the title column vs the purpose column. Is this repeated information?
for i in range(20):
    print(f'Row {i} -- title: {df["title"].iloc[i]} --- purpose: {df["purpose"].iloc[i]}')
print('____________________\n____________________')
for i in range(34989, 35011):
    print(f'Row {i} -- title: {df["title"].iloc[i]} --- purpose: {df["purpose"].iloc[i]}')
print('____________________\n____________________')
print(df['title'].unique())
print(df['purpose'].unique())
print('____________________\n____________________')

# The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.
df.drop('title', axis = 1, inplace = True)

# Create a value_counts of the mort_acc column.
print(df['mort_acc'].value_counts())
print('____________________\n____________________')

# Let's review the other columsn to see which most highly correlates to mort_acc
print(df.select_dtypes(include = ['number']).corr()['mort_acc'].sort_values())
print('____________________\n____________________')

# Looks like the total_acc feature correlates with the mort_acc, group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry.
mort_acc_mean = df.groupby('total_acc')['mort_acc'].mean()
print(mort_acc_mean)
print('____________________\n____________________')

# Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above.
mean_filled_df = df.copy()
mean_filled_df['mort_acc'] = mean_filled_df[['mort_acc', 'total_acc']].apply(lambda i: mort_acc_mean[i['total_acc']].round() if pd.isna(i['mort_acc']) else i['mort_acc'], axis = 1)
print(mean_filled_df.isna().sum())
print('____________________\n____________________')
print(mean_filled_df['mort_acc'])
print('____________________\n____________________')

it_imp_df = df.copy()
it_imputer = IterativeImputer(missing_values = np.nan, estimator = KNeighborsRegressor(n_neighbors = 3))
# it_imputer = IterativeImputer(missing_values = np.nan)
most_corr_cols = it_imp_df[['installment', 'revol_bal', 'loan_amnt', 'annual_inc', 'total_acc', 'mort_acc']]
it_imputer.set_output(transform = 'pandas')
imputed_cols = it_imputer.fit_transform(most_corr_cols)
imputed_values = imputed_cols['mort_acc'].round()
it_imp_df.fillna({'mort_acc' : imputed_values}, inplace = True)
print(it_imp_df.isna().sum())
print('____________________\n____________________')
print(it_imp_df['mort_acc'])
print('____________________\n____________________')

def unique_perc(ser: pd.Series):
    uniques = sorted(ser.unique())
    values = ser.value_counts()
    total = ser.count()
    percs = [values[i] / (total / 100) for i in uniques]
    if len(percs) < 34:
        percs.append(0)
    return percs


sns.barplot(unique_perc(df['mort_acc'].dropna()), color = 'lightsalmon', alpha = 0.6, edgecolor = 'dimgrey', ax = axes01[0, 0])
sns.barplot(unique_perc(mean_filled_df['mort_acc']), color = 'lightsteelblue', alpha = 0.6, edgecolor = 'dimgrey', ax = axes01[0, 0])

om_elements = [plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'lightsalmon', markersize = 8, label = 'Original'),
               plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'lightsteelblue', markersize = 8, label = 'Mean')]

axes01[0, 0].set_xlabel('Nº Accounts')
axes01[0, 0].set_yticks(range(41))
axes01[0, 0].set_ylabel('%')
axes01[0, 0].grid(True)
axes01[0, 0].set_title(f'Original / Mean Filled % distribution')
axes01[0, 0].legend(handles = om_elements, loc = 'upper right')


sns.barplot(unique_perc(df['mort_acc'].dropna()), color = 'lightsalmon', alpha = 0.6, edgecolor = 'dimgrey', ax =axes01[0, 1])
sns.barplot(unique_perc(it_imp_df['mort_acc']), color = 'greenyellow', alpha = 0.6, edgecolor = 'dimgrey', ax =axes01[0, 1])

im_elements = [plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'lightsalmon', markersize = 8, label = 'Original'),
               plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'greenyellow', markersize = 8, label = 'Iterative')]

axes01[0, 1].set_yticks(range(41))
axes01[0, 1].set_xlabel('Nº Accounts')
axes01[0, 1].set_ylabel('%')
axes01[0, 1].grid(True)
axes01[0, 1].set_title(f'Original / Iterative imputer % distribution')
axes01[0, 1].legend(handles = im_elements, loc = 'upper right')

##
## Sequential Model for Missing Values
##

# resampled_df = most_corr_cols[most_corr_cols['mort_acc'].between(0, 10).dropna()]

# df_to_fit = pd.DataFrame(columns = resampled_df.columns)
# for i in resampled_df['mort_acc'].unique():
#     k = resampled_df[resampled_df['mort_acc'] == i]
#     k = k.loc[k['mort_acc'].sample(800, replace = False).index]
#     df_to_fit = pd.concat([df_to_fit, k])

# print(df_to_fit['mort_acc'].nunique())
# print(df_to_fit.info())
# print(df_to_fit)
# print('____________________\n____________________')

# # mac_X = most_corr_cols.drop('mort_acc', axis = 1).dropna()[:358000]
# # mac_y = most_corr_cols['mort_acc'].dropna()[:358000]
# # mac_y = to_categorical(mac_y)

# mac_X = df_to_fit.drop('mort_acc', axis = 1)
# mac_y = df_to_fit['mort_acc']
# mac_y = to_categorical(mac_y)

# mac_X_train, mac_X_test, mac_y_train, mac_y_test = train_test_split(mac_X, mac_y, test_size = 0.25, random_state = 101)
# mac_scaler = MinMaxScaler()
# mac_X_train = mac_scaler.fit_transform(mac_X_train)
# mac_X_test = mac_scaler.fit_transform(mac_X_test)
# mac_model = Sequential()

# mac_model.add(Dense(128, activation = 'relu'))
# mac_model.add(Dense(64, activation = 'relu'))
# mac_model.add(Dense(32, activation = 'relu'))
# mac_model.add(Dense(16, activation = 'relu'))
# mac_model.add(Dense(11, activation = 'softmax'))
# mac_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['auc', 'precision'])

# mac_early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 50)
# mac_model.fit(x = mac_X_train, y = mac_y_train, epochs = 800, validation_data = (mac_X_test, mac_y_test), batch_size = 32, callbacks = [mac_early_stop])

# losses = pd.DataFrame(mac_model.history.history)

# mac_preds = mac_model.predict(mac_X_test)

# mac_pred_labels = np.argmax(mac_preds, axis = 1)
# mac_true_labels = np.argmax(mac_y_test, axis = 1)

# print(f'Classification Report:\n{classification_report(mac_true_labels, mac_pred_labels)}')
# print(f'Confusion Matrix:\n{confusion_matrix(mac_true_labels, mac_pred_labels)}')

# mac_model.save('Neural_Nets_and_Deep_Learning/mac_model.keras')
# ##

mac_df = df.copy()
mac_model = load_model('Neural_Nets_and_Deep_Learning/mac_model.keras')
mac_scaler = MinMaxScaler()
mac_scaler.fit(mac_df.select_dtypes(include = ['number']).dropna())

mac_df_to_fit = most_corr_cols[most_corr_cols['mort_acc'].isna()]
scaled_vals = mac_scaler.fit_transform(mac_df_to_fit.drop('mort_acc', axis = 1))
# scaled_vals_df = pd.DataFrame(data = scaled_vals, columns = mac_df_to_fit.drop('mort_acc', axis = 1).columns)
mac_df_to_fit['mort_acc'] = mac_model.predict(scaled_vals)
mac_df_to_fit['mort_acc'] = mac_df_to_fit['mort_acc'].round()

mac_df.fillna({'mort_acc' : mac_df_to_fit['mort_acc']}, inplace = True)
print(mac_df.isna().sum())
print('____________________\n____________________')
print(mac_df['mort_acc'])
print('____________________\n____________________')

sns.barplot(unique_perc(df['mort_acc'].dropna()), color = 'lightsalmon', alpha = 0.5, edgecolor = 'dimgrey', ax =axes01[1, 0])
sns.barplot(unique_perc(mac_df['mort_acc']), color = 'plum', alpha = 0.5, edgecolor = 'dimgrey', ax =axes01[1, 0])

im_elements = [plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'lightsalmon', markersize = 8, label = 'Original'),
               plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'plum', markersize = 8, label = 'Seq Model')]

axes01[1, 0].set_yticks(range(41))
axes01[1, 0].set_xlabel('Nº Accounts')
axes01[1, 0].set_ylabel('%')
axes01[1, 0].grid(True)
axes01[1, 0].set_title(f'Original / Sequential Model % distribution')
axes01[1, 0].legend(handles = im_elements, loc = 'upper right')

#The iterative imputer seems to be the method that maintains most of the orignal distribution.
df = it_imp_df
print(df.isna().sum())
print('____________________\n____________________')

# revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data.
# Go ahead and remove the rows that are missing those values in those columns with dropna().
df.dropna(inplace = True)
print(df.isna().sum())
print('____________________\n____________________')

# List all the columns that are currently non-numeric.
print(df.select_dtypes(include = ['object']).head(5))
print(df.select_dtypes(include = ['object']).columns)
print('____________________\n____________________')

# Convert the term feature into either a 36 or 60 integer numeric data type
df['term'] = df['term'].apply(lambda i: int(str(i[:3])))
print(df['term'])
print('____________________\n____________________')

# We already know grade is part of sub_grade, so just drop the grade feature.
df.drop('grade', axis = 1, inplace = True)

# Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe.
subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first = True)
df = pd.concat([df.drop('sub_grade', axis = 1), subgrade_dummies], axis = 1)


# Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe.
vs_at_ils_p_dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']], drop_first = True)
df = pd.concat([df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis = 1), vs_at_ils_p_dummies], axis = 1)

# Review the value_counts for the home_ownership column.
print(df['home_ownership'].value_counts())
print('____________________\n____________________')

# Convert these to dummy variables, but [replace] NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe
df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
ho_dummies = pd.get_dummies(df['home_ownership'], drop_first = True)
df = pd.concat([df.drop('home_ownership', axis = 1), ho_dummies], axis = 1)

# Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.
df['zip_code'] = df['address'].apply(lambda i: i[-5:])

# Now make this zip_code column into dummy variables using pandas. Concatenate the result and drop the original zip_code column along with dropping the address column.
zip_c_dummies = pd.get_dummies(df['zip_code'], drop_first = True)
df = pd.concat([df.drop(['zip_code', 'address'], axis = 1), zip_c_dummies], axis = 1)
print(df.columns)
print('____________________\n____________________')

# issue_d 
# This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.
df.drop('issue_d', axis = 1, inplace = True)

# earliest_cr_line
# This appears to be a historical time stamp feature. Extract the year from this feature then convert it to a numeric feature.
# Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda i: int(i[-4:]))
df.drop('earliest_cr_line', axis = 1, inplace = True)

# Drop the loan_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.
df.drop('loan_status', axis = 1, inplace = True)
print('____________________\n____________________')

print(df.select_dtypes(include = ['object']).columns)


#Saved resulting csv for use in the model module
pd.DataFrame.to_csv(df, 'Neural_Nets_and_Deep_Learning/lending_club_loan_engineered.csv')


plt.show()
