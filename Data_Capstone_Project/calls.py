'''
911 Calls Capstone Project
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Read in the csv file as a dataframe called df 
df = pd.read_csv('Data_Capstone_Project\\911.csv')

# Check the info() of the df
print(df.info())

# Check the head of df
print(df.head())

# What are the top 5 zipcodes for 911 calls?
print(df['zip'].value_counts().head(5))

# What are the top 5 townships (twp) for 911 calls?
print(df['twp'].value_counts().iloc[:5])

# Take a look at the 'title' column, how many unique title codes are there?
print(df['title'].nunique())

##
## Creating new features
##

#  In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic.
#  Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
#  For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. 
df['Reason'] = df['title'].apply(lambda i: i.split(':')[0])

# What is the most common Reason for a 911 call based off of this new column?
print(df['Reason'].value_counts().idxmax())

fig, axes = plt.subplots(2, 3, figsize = (18, 9))
plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.97, bottom = 0.04)

# Use seaborn to create a countplot of 911 calls by Reason.
sns.countplot(x = 'Reason', data = df, hue = 'Reason', palette = 'BuPu', ax = axes[0, 0])
axes[0, 0].set_title('Reason')

# What is the data type of the objects in the timeStamp column?
print(df['timeStamp'].dtype)

# Use [pd.to_datetime] to convert the column from strings to DateTime objects.
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week.
# You will create these columns based off of the timeStamp column.
df['Hour'] = df['timeStamp'].dt.hour
df['Day'] = df['timeStamp'].dt.day_of_week
df['Month'] = df['timeStamp'].dt.month

# df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
# df['Day'] = df['timeStamp'].apply(lambda x: x.day_of_week)
# df['Month'] = df['timeStamp'].apply(lambda x: x.month)

# Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: 
# dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
dmap = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['Day'] = df['Day'].map(dmap)
# df['Day'] = df['Day'].apply(lambda i: dmap[i])

# Use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.
sns.countplot(df, x = 'Day', hue = 'Reason', palette = 'flare', ax = axes[0, 1])
axes[0, 1].set_title('Day/Reason')

# Now do the same for Month:
sns.countplot(df, x = 'Month', hue = 'Reason', palette = 'PuOr', ax = axes[0, 2])
axes[0, 2].set_title('Month/Reason')
# plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)

# Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.
print(df.groupby('Month').count().head())

# Now create a simple plot off of the dataframe indicating the count of calls per month. 
sns.lineplot(x = df['Month'].value_counts().index, y = df['Month'].value_counts().values, ax = axes[1, 0])
axes[1, 0].set_title('Calls by Month')

# use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column.
lmdf = df['Month'].value_counts().reset_index()
lmdf.columns = ['Month', 'counts']
sns.lmplot(x = 'Month', y = 'counts', data = lmdf)

# Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.
df['Date'] = df['timeStamp'].dt.date

# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
sns.lineplot(data = df.groupby('Date').count()['lat'], ax = axes[1, 1])
axes[1, 0].set_ylabel('Nº Calls')
axes[1, 0].set_title('Date Counts')

# dcdf = df['Date'].value_counts().reset_index()
# dcdf.columns = ['Date', 'counts']
# sns.lineplot(dcdf, x = 'Date', y = 'counts')

# Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call
# df[df['Reason'] == 'Traffic'].groupby('Date').count()['lat'].plot()
dcdf = df.groupby(['Date', 'Reason']).size().reset_index(name = 'counts')
dcdf = dcdf.pivot_table(index = 'Date', columns = 'Reason', values = 'counts').reset_index()
fig02, axes02 = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 4))
sns.lineplot(dcdf, x = 'Date', y = 'EMS', ax = axes02[0])
sns.lineplot(dcdf, x = 'Date', y = 'Fire', ax = axes02[1])
sns.lineplot(dcdf, x = 'Date', y = 'Traffic', ax = axes02[2])

# Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week.
hmdf = df.groupby(by = ['Day', 'Hour']).count()['Reason'].unstack()
hmdf = df.pivot_table(index = 'Day', columns = 'Hour', aggfunc = 'size')

# Now create a HeatMap using this new DataFrame.
sns.heatmap(hmdf, cmap = 'Spectral', ax = axes[1, 2])

# Now create a clustermap using this DataFrame.
sns.clustermap(hmdf, cmap = 'magma')

#Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.
hmdf = df.pivot_table(index = 'Day', columns = 'Month', aggfunc = 'size')
sns.heatmap(hmdf, cmap = 'Spectral', ax = axes[1, 1])
sns.clustermap(hmdf, cmap = 'magma')

plt.show()