'''
Seaborn Exercises
The Data
We will be working with a famous titanic data set for these exercises
import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset('titanic')
sns.set_style('whitegrid')
print(titanic.head())
'''

import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

sns.set_style('whitegrid')

print(titanic.head())

# Recreate the plots below using the titanic dataframe.
fig, axes = plt.subplots(2, 3, figsize = (18, 9))

sns.jointplot(x = 'fare', y = 'age', data = titanic)

sns.histplot(titanic['fare'], color = 'red', bins = 30, ax = axes[0, 0])

sns.boxplot(x = 'class', y = 'age', data = titanic, hue = 'class', palette = 'rainbow', ax = axes[0, 1])

sns.swarmplot(x = 'class', y = 'age', data = titanic, hue = 'class', palette = 'Set2', ax = axes[1, 1])
axes[1, 1].set_position([0.4, 0.05, 0.6, 0.4])
axes[1, 2].remove()

sns.countplot(x = 'sex', data = titanic, ax = axes[0, 2])

tc = titanic.corr(numeric_only = True)
sns.heatmap(tc, cmap = 'coolwarm', ax = axes[1, 0])
plt.title('titanic')

tg = sns.FacetGrid(data = titanic, col = 'sex')
tg.map(sns.histplot, 'age')
tg.map(plt.hist, 'age')


plt.show()