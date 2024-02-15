'''
Choropleth Maps Exercise
'''

import plotly.graph_objs as go
import pycountry

# Import pandas and read the csv file: 2014_World_Power_Consumption
import pandas as pd
pc_df = pd.read_csv('Geographical_Plotting\\2014_World_Power_Consumption')

# Check the head of the DataFrame.
print(pc_df.head())

# Create a Choropleth Plot of the Power Consumption for Countries using the data and layout dictionary.
data = {'type' : 'choropleth',
        'locationmode' : 'country names',
        'locations' : pc_df['Country'],
        'colorscale' : 'Cividis',
        'text' : pc_df['Text'],
        'z' : pc_df['Power Consumption KWH'],
        'colorbar' : {'title' : 'Power Consumption KWH'},
        'marker' : {'line' : {'color' : 'rgb(0, 0, 0)', 'width' : 1}}}

layout = {'title' : 'Power Consumption',
          'geo' : {'showframe' : False,
                   'projection' : {'type' : 'equal earth'}}}

choromap01 = go.Figure(data = [data], layout = layout)
choromap01.show()

# Import the 2012_Election_Data csv file using pandas.
ed_df = pd.read_csv('Geographical_Plotting\\2012_Election_Data')

# Check the head of the DataFrame.
print(ed_df.head())


# Create a plot that displays the Voting-Age Population (VAP) per state.
data = {'type' : 'choropleth',
        'locations' : ed_df['State Abv'],
        'locationmode' : 'USA-states',
        'colorscale' : 'ylorrd',
        'text' : [ed_df['State'], ed_df['Voting-Age Population (VAP)']],
        'z' : ed_df['Voting-Age Population (VAP)'],
        'colorbar' : {'title' : 'VAP'},
        'marker' : {'line' : {'color' : 'rgb(0, 0, 0)', 'width' : 1}}}

layout = {'title' : 'Voting-Age Population',
          'geo' : {'scope' : 'usa', 'showlakes' : True, 'lakecolor' : 'rgb(85, 173, 240)'}}

choromap02 = go.Figure(data = [data], layout = layout)

choromap02.show()





