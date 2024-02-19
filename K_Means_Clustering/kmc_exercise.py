'''
K Means Clustering Project 
For this project we will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

# Read in the College_Data file using read_csv. Figure out how to set the first column as the index.
df = pd.read_csv('K_Means_Clustering/College_Data', index_col = 0)

# Check the head of the data.
print(df.head(), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')

# Check the info() and describe() methods on the data.
print(df.info(), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(df.describe(), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')

fig, axes = plt.subplots(2, 3, figsize = (18, 9))
plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.96, bottom = 0.06)
axes[1, 2].remove()

# Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column.
sns.scatterplot(df, x ='Room.Board', y = 'Grad.Rate', hue = 'Private', palette = 'gist_ncar', ax = axes[0, 0])
axes[0, 0].set_title('Grad Rate vs Room Board')
axes[0, 0].grid(True)

# Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.
sns.scatterplot(df, x = 'Outstate', y = 'F.Undergrad', hue = 'Private', palette = 'hsv', ax = axes[0, 1])
axes[0, 1].set_title('F Undergrad vs Outstate')
axes[0, 1].grid(True)

# Create a stacked histogram showing Out of State Tuition based on the Private column.
sns.histplot(df, x = 'Outstate', hue = 'Private', palette = 'YlOrRd', bins = 45, ax = axes[0, 2])
axes[0, 2].set_title('Outstate / Private')
axes[0, 2].grid(True)
axes[0, 2].set_xticks(range(0, 25000, 2500))
axes[0, 2].set_yticks(range(0, 40, 2))

# Create a similar histogram for the Grad.Rate column.
sns.histplot(df, x = 'Grad.Rate', hue = 'Private', palette = 'YlGnBu', bins = 45, ax = axes[1, 0])
axes[1, 0].set_position([0.04, 0.06, 0.44, 0.41])
axes[1, 0].set_title('Grad Rate / Private')
axes[1, 0].grid(True)
axes[1, 0].set_xticks(range(0, 100, 5))
axes[1, 0].set_yticks(range(0, 46, 2))

# Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?
print(df[df['Grad.Rate'] > 100], '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')

# Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error) when doing this operation, so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.
df.loc['Cazenovia College', 'Grad.Rate'] = 100
print(df[df['Grad.Rate'] > 100], '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
sns.histplot(df, x = 'Grad.Rate', hue = 'Private', palette = 'BuPu', bins = 45, ax = axes[1, 1])
axes[1, 1].set_position([0.55, 0.06, 0.44, 0.41])
axes[1, 1].set_title('Grad Rate / Private')
axes[1, 1].grid(True)
axes[1, 1].set_xticks(range(0, 100, 5))
axes[1, 1].set_yticks(range(0, 38, 2))

# Create an instance of a K Means model with 2 clusters.
kmeans = KMeans(n_clusters = 2)

# Fit the model to all the data except for the Private label.
kmeans.fit(df.drop('Private', axis = 1))

# What are the cluster center vectors?
print(kmeans.cluster_centers_, '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')

# Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.
df['Cluster'] = df['Private'].apply(lambda i: 1 if i == 'Yes' else 0)

# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.
print(confusion_matrix(df['Cluster'], kmeans.labels_), '\n-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -')
print(classification_report(df['Cluster'], kmeans.labels_))

plt.show()