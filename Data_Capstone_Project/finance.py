'''
Finance Data Project 
'''

import numpy as np
import pandas as pd
from pandas_datareader import DataReader
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import cufflinks as cf 
from plotly.offline import iplot
import time

bsdf = pd.read_pickle('Data_Capstone_Project\\all_banks')

# Check the head of the bank_stocks dataframe.
print(bsdf.head())

# Create a list of the ticker symbols (as strings) in alphabetical order. Call this list: tickers
tickers = bsdf.columns.get_level_values('Bank Ticker').unique().to_list()

# What is the max Close price for each bank's stock throughout the time period?
print(bsdf.xs(key = 'Close', axis = 1, level = 'Stock Info').max())
print(bsdf.xs(key = 'Close', axis = 1, level = 1).max())

for tick in tickers:
    print(tick, bsdf[tick]['Close'].max())

# Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock.
# returns = pd.DataFrame()

#We can use pandas pct_change() method on the Close column to create a column representing this return value.
# Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.*
# for t in tickers:
#     returns[t + ' Return'] = bsdf[tick]['Close'].pct_change()

returns = bsdf.xs(key = 'Close', axis = 1, level = 1).pct_change()
for t in tickers:
    returns.rename(columns = {t : t + ' Return'}, inplace = True)

# Create a pairplot using seaborn of the returns dataframe.
sns.pairplot(returns, height = 2)

# Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. 
max_min_df = pd.DataFrame({'Max Value' : returns.max(), 'Max Date' : returns.idxmax(),
                           'Min Value' : returns.min(), 'Min Date' : returns.idxmin()})
print(max_min_df.head())

# Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?
print(returns.std())
print(returns.loc['2015-01-01' : '2015-12-31'].std().max())
print(returns.std().max())

fig, axes = plt.subplots(2, 3, figsize = (14, 9))

# Create a distplot using seaborn of the 2015 returns for Morgan Stanley
sns.histplot(returns['MS Return'].loc['2015-01-01' : '2015-12-31'], color = '#786558', kde = True, bins = 50, ax = axes[0, 0])
axes[0, 0].set_title('Morgan Stanley 2015 returns')
# sns.displot(returns['MS Return'].loc['2015-01-01' : '2015-12-31'], color = 'green', kde = True, bins = 50)

# Create a distplot using seaborn of the 2008 returns for CitiGroup
sns.histplot(returns['C Return'].loc['2008-01-01' : '2008-12-31'], color = 'green', kde = True, bins = 50, ax = axes[0, 1])
axes[0, 1].set_title('CitiGroup 2008 returns')
# sns.displot(returns['C Return'][(returns.index > '2007-12-31') & (returns.index < '2009-01-01')], color = '#125888', kde = True, bins = 50)

# Create a line plot showing Close price for each bank for the entire index of time.
sns.lineplot(bsdf.xs(key = 'Close', axis = 1, level = 1), ax = axes[0, 2])
axes[0, 2].set_title('Close prices')

# Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008
# avgdf = pd.DataFrame({'Rol Mean' : bsdf.loc[:, ('BAC', 'Close')][(bsdf.loc[:, ('BAC', 'Close')].index > '2007-12-31') & (bsdf.loc[:, ('BAC', 'Close')].index < '2009-01-01')].rolling(window = 30).mean(),
#                       'Close' : bsdf.loc[:, ('BAC', 'Close')][(bsdf.loc[:, ('BAC', 'Close')].index > '2007-12-31') & (bsdf.loc[:, ('BAC', 'Close')].index < '2009-01-01')]})
avgdf = pd.DataFrame({'Rol Mean' : bsdf.loc[:, ('BAC', 'Close')].loc['2008-01-01' : '2008-12-31'].rolling(window = 30).mean(),
                      'Close' : bsdf.loc[:, ('BAC', 'Close')].loc['2008-01-01' : '2008-12-31']})
sns.lineplot(avgdf, palette = 'cubehelix', ax = axes[1, 0])
axes[1, 0].set_title('BAC rol 30 day avg 2008')

# Create a heatmap of the correlation between the stocks Close Price.
cordf = bsdf.xs(key = 'Close', axis = 1, level = 1).corr()
sns.heatmap(cordf, annot = True, cmap = 'coolwarm', ax = axes[1, 1])
axes[1, 1].set_title('Close price correlation')

# Use seaborn's clustermap to cluster the correlations together:
sns.clustermap(cordf, annot = True, cmap = 'coolwarm')

# Use .iplot(kind='candle) to create a candle plot of Bank of America's stock from Jan 1st 2015 to Jan 1st 2016.
# fig01 = bsdf['BAC'][(bsdf['BAC'].index > '2014-12-31') & (bsdf['BAC'].index < '2016-01-02')].iplot(kind = 'candle', asFigure = True)
fig01 = bsdf['BAC'][['Open', 'High', 'Low', 'Close']].loc['2015-01-01' : '2015-12-31'].iplot(kind = 'candle', asFigure = True)

# Use .ta_plot(study='sma') to create a Simple Moving Averages plot of Morgan Stanley for the year 2015.
# fig02 = bsdf['MS'][(bsdf['MS'].index > '2014-12-31') & (bsdf['MS'].index < '2016-01-01')]['Close'].ta_plot(study = 'sma', asFigure = True)
fig02 = bsdf['MS'].loc['2015-01-01' : '2015-12-31']['Close'].ta_plot(study = 'sma', asFigure = True)

# Use .ta_plot(study='boll') to create a Bollinger Band Plot for Bank of America for the year 2015.
# fig03 = bsdf['BAC'][(bsdf['BAC'].index > '2014-12-31') & (bsdf['BAC'].index < '2016-01-01')].ta_plot(study = 'boll', asFigure = True)
fig03 = bsdf['BAC'].loc['2015-01-01' : '2015-12-31'].ta_plot(study = 'boll', asFigure = True)

iplot(fig01)
iplot(fig02)
iplot(fig03)
plt.show()