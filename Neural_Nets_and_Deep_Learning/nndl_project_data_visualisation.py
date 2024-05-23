'''
Keras API Project Exercise

Goal

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off),
can we build a model thatcan predict wether or nor a borrower will pay back their loan?
This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan.

The "loan_status" column contains our label.



Data Visualisation Section
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the data
df = pd.read_csv('Neural_Nets_and_Deep_Learning/lending_club_loan_two.csv')
print(df.head())
print('____________________\n____________________')
print(df.info())
print('____________________\n____________________')
print(df.describe().transpose)
print('____________________\n____________________')

###
fig01, axes01 = plt.subplots(2, 2, figsize = (18, 9))
# plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.96, bottom = 0.06)

fig02, axes02 = plt.subplots(2, 2, figsize = (18, 9))
# plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.96, bottom = 0.06)

fig03, axes03 = plt.subplots(2, 2, figsize = (18, 9))
# plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.96, bottom = 0.06)

fig04, axes04 = plt.subplots(figsize = (8, 4))
plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.96, bottom = 0.06)

# Create a countplot of loan_status
sns.countplot(df, x = 'loan_status', hue = 'loan_status', palette = ['indigo', 'olive'], ax = axes01[0, 0])
axes01[0, 0].set_yticks(range(0, 340000, 20000))
axes01[0, 0].grid(True)
axes01[0, 0].set_title('Loan Status Count')

# Create a histogram of the loan_amnt column.
sns.histplot(df['loan_amnt'], bins = 35, color = 'violet', ax = axes01[0, 1])
axes01[0, 1].set_xticks(range(0, 45000, 5000))
axes01[0, 1].set_yticks(range(0, 42000, 2000))
axes01[0, 1].grid(True)
axes01[0, 1].set_title('Loan Amount Count')

# Calculate the correlation between all continuous numeric variables using .corr() method.
print(df.select_dtypes(include = ['number']).corr())
print('____________________\n____________________')

# Visualize this using a heatmap.
sns.heatmap(df.select_dtypes(include = ['number']).corr(), annot = True, cmap = 'BuPu', ax = axes01[1, 0])
axes01[1, 0].set_position([0.06, 0.1, 0.34, 0.37])
axes01[1, 0].set_title('Correlations Heatmap')

# You should have noticed almost perfect correlation with the "installment" feature. Explore this feature further. Perform a scatterplot between them.
sns.scatterplot(df, x = 'installment', y = 'loan_amnt', color = 'deepskyblue', ax = axes01[1, 1])
axes01[1, 1].set_yticks(range(0, 42000, 2000))
axes01[1, 1].grid(True)
axes01[1, 1].set_title('Installment vs Amount Scatterplot')

# Create a boxplot showing the relationship between the loan_status and the Loan Amount.
sns.boxplot(df, x = 'loan_status', y = 'loan_amnt', hue = 'loan_status', palette = ['green', 'peru'], ax = axes02[0, 0])
axes02[0, 0].set_title('Loan Status vs Loan Amount Boxplot')

# Calculate the summary statistics for the loan amount, grouped by the loan_status.
print(df.groupby('loan_status')['loan_amnt'].describe())
print('____________________\n____________________')

# What are the unique possible grades and subgrades?
print(f'Grades :  {df['grade'].unique()}')
print(f'Subgrades :  {df['sub_grade'].unique()}')
print('____________________\n____________________')

# Create a countplot per grade. Set the hue to the loan_status label.
sns.countplot(df, x = 'grade', hue = 'loan_status', palette = ['darkcyan', 'salmon'], ax = axes02[0, 1])
axes02[0, 1].set_yticks(range(0, 110000, 10000))
axes02[0, 1].grid(True)
axes02[0, 1].set_title('Loan Status Grades Count')

# Display a count plot per subgrade. Explore both all loans made per subgrade as well being separated based on the loan_status. After creating this plot, go ahead and create a similar plot, but set hue="loan_status"
sns.countplot(df, x = 'sub_grade', hue = 'sub_grade', palette = 'viridis', order = df['sub_grade'].sort_values().unique(), ax = axes02[1, 0])
axes02[1, 0].set_yticks(range(0, 30000, 2000))
axes02[1, 0].grid(True)
axes02[1, 0].set_title('Sub-Grade Counts')

sns.countplot(df, x = 'sub_grade', hue = 'loan_status', order = df['sub_grade'].sort_values().unique(), palette = ['darkblue', 'limegreen'], ax = axes02[1, 1])
axes02[1, 1].set_yticks(range(0, 26000, 2000))
axes02[1, 1].grid(True)
axes02[1, 1].set_title('Sub-Grade paired with Loan Status Counts')

# It looks like F and G subgrades don't get paid back that often. Isloate those and recreate the countplot just for those subgrades.
sns.countplot(df[(df['grade'] == 'F') | (df['grade'] == 'G')],
              x = 'sub_grade', hue = 'loan_status',
              order = df['sub_grade'][(df['grade'] == 'F') | (df['grade'] == 'G')].sort_values().unique(),
              palette = ['powderblue', 'greenyellow'], ax = axes03[0, 0])
axes03[0, 0].set_yticks(range(0, 2400, 100))
axes03[0, 0].grid(True)
axes03[0, 0].set_title('Sub-Grade F and G paired with Loan Status Counts')

# Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".
df['loan_repaid'] = df['loan_status'].apply(lambda i: 1 if i == 'Fully Paid' else 0)
print(df[['loan_repaid', 'loan_status']])
print('____________________\n____________________')

# Create a bar plot showing the correlation of the numeric features to the new loan_repaid column. 
sns.barplot(df.select_dtypes(include = ['number']).corr()['loan_repaid'][:-1].sort_values(), color = 'blueviolet', ax = axes03[0, 1])
axes03[0, 1].set_position([0.54, 0.62, 0.43, 0.35])
axes03[0, 1].tick_params(axis = 'x', rotation=90)
axes03[0, 1].set_title('Correlations with Loan Repaid')

# What is the length of the dataframe?
print(len(df))
print('____________________\n____________________')

# Create a Series that displays the total count of missing values per column.
print(df.isna().sum())
print('____________________\n____________________')

# Convert this Series to be in term of percentage of the total DataFrame
print(df.isna().sum().apply(lambda i: 0 if i == 0 else i / (len(df) / 100)))
print('____________________\n____________________')

# How many unique employment job titles are there?
print(df['emp_title'].nunique())
print('____________________\n____________________')

# Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column
df.drop('emp_title', axis = 1, inplace = True)

# Create a count plot of the emp_length feature column.
lenghts = pd.Series(df['emp_length'].sort_values().unique())
sorted_lenghts = lenghts.dropna()
changes = [0, 1, len(sorted_lenghts) - 1]
sorted_lenghts[changes] = lenghts[len(lenghts) - 2], lenghts[0], lenghts[1]

sns.countplot(df, x = 'emp_length', order = sorted_lenghts, hue = 'emp_length', palette = 'Set3', ax = axes03[1, 0])
axes03[1, 0].set_yticks(range(0, 140000, 10000))
axes03[1, 0].grid(True)
axes03[1, 0].set_title('Employment Length Count')

# Plot out the countplot with a hue separating Fully Paid vs Charged Off
sns.countplot(df, x = 'emp_length', order = sorted_lenghts, hue = 'loan_status', palette = 'rocket', ax = axes03[1, 1])
axes03[1, 1].set_yticks(range(0, 120000, 10000))
axes03[1, 1].grid(True)
axes03[1, 1].legend(loc = 'upper left')
axes03[1, 1].set_title('Employment Length Count (Fully Paid/Charged Off)')

# This still doesn't really inform us if there is a strong relationship between employment length and being charged off, what we want is the percentage of charge offs per category.
# Essentially informing us what percent of people per employment category didn't pay back their loan. 
perc_co_el = pd.DataFrame(data = sorted_lenghts.values, columns = ['Emp Length'])
perc_co_el['Percs'] = perc_co_el['Emp Length'].apply(lambda i: df['emp_length'][(df['emp_length'] == i) & (df['loan_repaid'] == 0)].count() / ((df['emp_length'][df['emp_length'] == i].count()) / 100))
print(perc_co_el)
print('____________________\n____________________')

sns.barplot(perc_co_el, x = 'Emp Length', y = 'Percs', color = 'goldenrod', ax = axes04)
axes04.set_yticks(range(0, 22))
axes04.grid(True)
axes04.set_title(f'% of Charged Off for Emplyment Length')

# Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column.
df.drop('emp_length', axis = 1, inplace = True)

print(df)
print('____________________\n____________________')
print(f'{df.shape}\n{df.info()}')

plt.show()