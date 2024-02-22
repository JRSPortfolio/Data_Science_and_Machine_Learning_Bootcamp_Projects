'''
Natural Language Processing Project
In this NLP project you will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# Read the yelp.csv file and set it as a dataframe called yelp.
yelp = pd.read_csv('Natural_Language_Processing/yelp.csv')

# Check the head, info , and describe methods on yelp.
print(yelp.head())
print('-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ')
print(yelp.info())
print('-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ')
print(yelp.describe())
print('-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ')

# Create a new column called "text length" which is the number of words in the text column.
def count_words(text: str):
    rem_punc = [c for c in text if c not in string.punctuation]
    rem_punc = ''.join(rem_punc)
    return len(rem_punc.split())

yelp['text_lenght'] = yelp['text'].apply(count_words)

# create a grid of 5 histograms of text length based off of the star ratings.
fg = sns.FacetGrid(yelp, col = 'stars', hue = 'stars', palette = 'Dark2', height = 3.5)
fg.map(plt.hist, 'text_lenght', bins = 40, edgecolor = 'black', linewidth = 0.2)
fg.set_xlabels('Nº Words')
fg.set(xticks = range(0, 900, 100), yticks = range(0, 550, 50))
for i in range(5):
    fg.axes[0, i].grid(True)

# Create a boxplot of text length for each star category.
fig, axes = plt.subplots(1, 3, figsize = (18, 6))
plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.99, bottom = 0.06)

sns.boxplot(yelp, x = 'stars', y = 'text_lenght', hue = 'stars', palette = 'gist_rainbow_r', ax = axes[0])
axes[0].set_xlabel('Stars')
axes[0].set_ylabel('Nº Words')

# Create a countplot of the number of occurrences for each type of star rating.
sns.countplot(yelp, x = 'stars', hue = 'stars', palette = 'brg', ax = axes[1])
axes[1].set_xlabel('Stars')
axes[1].set_yticks(range(0, 3700, 200))
axes[1].grid(True)

# Use groupby to get the mean values of the numerical columns.
gb_stars = yelp.select_dtypes(include = 'number').groupby('stars')
print(gb_stars.mean())
print('-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ')

# Use the corr() method on that groupby dataframe to produce this dataframe:
corr_db_stars = gb_stars.mean().corr()
print(corr_db_stars)
print('-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ')

# Then use seaborn to create a heatmap based off that .corr() dataframe.
sns.heatmap(corr_db_stars, cmap = 'coolwarm', annot = True, ax = axes[2])

# Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

# Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)
X01 = yelp_class['text']
y01 = yelp_class['stars']

# Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.
X01 = CountVectorizer().fit_transform(X01)

# Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101
X_train01, X_test01, y_train01, y_test01 = train_test_split(X01, y01, test_size = 0.3, random_state = 101)

# Create an instance of the estimator and call is nb
nb = MultinomialNB()

# Fit nb using the training data.
nb.fit(X_train01, y_train01)

# Use the predict method off of nb to predict labels from X_test.
preds01 = nb.predict(X_test01)

# Create a confusion matrix and classification report using these predictions and y_test.
print(confusion_matrix(y_test01, preds01))
print(classification_report(y_test01, preds01))
print('-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ')

# Create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()
pipeline = Pipeline([('bow', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())])

# Redo the train test split on the yelp_class object.
X02 = yelp_class['text']
y02 = yelp_class['stars']

X_train02, X_test02, y_train02, y_test02 = train_test_split(X02, y02, test_size = 0.3, random_state = 101)

# Fit the pipeline to the training data.
pipeline.fit(X_train02, y_train02)

# Use the pipeline to predict from the X_test and create a classification report and confusion matrix.
preds02 = pipeline.predict(X_test02)

print(confusion_matrix(y_test02, preds02))
print(classification_report(y_test02, preds02))
print('-    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ')

plt.show()