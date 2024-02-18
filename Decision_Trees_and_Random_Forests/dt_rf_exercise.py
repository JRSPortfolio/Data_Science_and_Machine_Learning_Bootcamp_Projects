'''
Random Forest Project

Here are what the columns represent:
* credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
* purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
* int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
* installment: The monthly installments owed by the borrower if the loan is funded.
* log.annual.inc: The natural log of the self-reported annual income of the borrower.
* dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
* fico: The FICO credit score of the borrower.
* days.with.cr.line: The number of days the borrower has had a credit line.
* revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
* revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
* inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
* delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
* pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Use pandas to read loan_data.csv as a dataframe called loans.
loans = pd.read_csv('Decision_Trees_and_Random_Forests/loan_data.csv')

# Check out the info(), head(), and describe() methods on loans.
print(loans.head(), '\n-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(loans.info(), '\n-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(loans.describe(), '\n-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')


fig, axes = plt.subplots(3, 1, figsize = (18, 9))
plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.99, bottom = 0.06)

# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.
sns.histplot(loans, x = 'fico', hue = 'credit.policy', palette = 'cividis', ax = axes[0], bins = 50)
axes[0].legend('10').set_title('Credit Policy')
axes[0].grid(True)
axes[0].set_xticks(range(600, 860, 10))
axes[0].set_yticks(range(0, 550, 50))

# Create a similar figure, except this time select by the not.fully.paid column.
sns.histplot(loans, x = 'fico', hue = 'not.fully.paid', palette = 'Paired', ax = axes[1], bins = 50)
axes[1].legend('10').set_title('Not Fully Paid')
axes[1].grid(True)
axes[1].set_xticks(range(600, 860, 10))
axes[1].set_yticks(range(0, 550, 50))

# Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
sns.countplot(loans, x = 'purpose', hue = 'not.fully.paid', palette = 'PuOr', ax = axes[2])
axes[2].legend('10').set_title('Not Fully Paid')
axes[2].grid(True)
axes[2].set_yticks(range(0, 3750, 250))

# Let's see the trend between FICO score and interest rate. Recreate the following jointplot.
sns.jointplot(loans, x = 'fico', y = 'int.rate', color = 'orchid')

# Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy.
lmp = sns.lmplot(loans, x = 'fico', y = 'int.rate', hue = 'credit.policy', col = 'not.fully.paid', palette = ['darkorange', 'indigo'], height = 8)
lmp._legend.set_title('Credit Policy')
lmp.set_titles('Not Fully Paid - {col_name}')
lmp.set_xlabels('FICO')
lmp.set_ylabels('Interest Rate')

# Notice that the **purpose** column as categorical
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.
# cat_feats = ['purpose']

# Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.
final_data = pd.get_dummies(loans, columns = ['purpose'], drop_first = True)

# Use sklearn to split your data into a training set and a testing set as we've done in the past.
X = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Create predictions from the test set and create a classification report and a confusion matrix.
predictions = dtree.predict(X_test)
print(confusion_matrix(y_test, predictions), '\n-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(classification_report(y_test, predictions), '\n-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')

# Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.
rfc = RandomForestClassifier(n_estimators = 300)
rfc.fit(X_train, y_train)

# Let's predict off the y_test values and evaluate our model.
# Predict the class of not.fully.paid for the X_test data.
rfc_pred = rfc.predict(X_test)

# Now create a classification report from the results.
print(confusion_matrix(y_test, rfc_pred), '\n-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(classification_report(y_test, rfc_pred))

plt.show()