'''
Ecommerce Purchases Exercise
'''
import time
# Import pandas and read in the Ecommerce Purchases csv file and set it to a DataFrame called ecom. 
import pandas as pd
ecom = pd.read_csv("Pandas\\ECommerce Purchases")

# Check the head of the DataFrame
print(ecom.head())
print('------------------------------------------\n------------------------------------------')

# How many rows and columns are there?
print(ecom.shape)
print(ecom.info())
print('------------------------------------------\n------------------------------------------')

# What is the average Purchase Price?
print(f'Average Purchase Price: {ecom['Purchase Price'].mean()}')
print('------------------------------------------\n------------------------------------------')

# What were the highest and lowest purchase prices?
print(f'highest: {ecom['Purchase Price'].max()}\n lowest: {ecom['Purchase Price'].min()}')
print('------------------------------------------\n------------------------------------------')

# How many people have English 'en' as their Language of choice on the website?
print(ecom.loc[ecom['Language'] == 'en'].count())
print(ecom[ecom['Language'] == 'en']['Language'].count())
print('------------------------------------------\n------------------------------------------')

# How many people have the job title of "Lawyer"?
print(ecom.loc[ecom['Job'] == 'Lawyer'].count())
print(ecom['Job'][ecom['Job'].str.contains('lawyer', case = False)].count())
print(ecom[ecom['Job'] == 'Lawyer'].count())
print('------------------------------------------\n------------------------------------------')

# How many people made the purchase during the AM and how many people made the purchase during PM? 
print(ecom['AM or PM'].value_counts())
print('------------------------------------------\n------------------------------------------')

# What are the 5 most common Job Titles?
print(ecom['Job'].value_counts().head(5))
print(ecom['Job'].value_counts().iloc[0:5])
print('------------------------------------------\n------------------------------------------')

# Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction?
print(ecom['Purchase Price'].loc[ecom['Lot'] == '90 WT'].iloc[0])
print(ecom[ecom['Lot'] == '90 WT']['Purchase Price'])
print('------------------------------------------\n------------------------------------------')

# What is the email of the person with the following Credit Card Number: 4926535242672853?
print(ecom['Email'].loc[ecom['Credit Card'] == 4926535242672853])
print(ecom[ecom['Credit Card'] == 4926535242672853]['Email'])
print('------------------------------------------\n------------------------------------------')

# How many people have American Express as their Credit Card Provider *and* made a purchase above $95?
print(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)].count())
print('------------------------------------------\n------------------------------------------')

# How many people have a credit card that expires in 2025?
print(ecom[ecom['CC Exp Date'].str.split('/').str[1] == '25'].count())
print(ecom[ecom['CC Exp Date'].str.split('/').str[1] == '25'].count().iloc[0])
print(sum(ecom['CC Exp Date'].apply(lambda exp: exp[3:] == '25')))
print(ecom[ecom['CC Exp Date'].apply(lambda exp: exp[3:] == '25')].count())
print('------------------------------------------\n------------------------------------------')

# What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...)?
print(ecom['Email'].str.split('@').str.get(1).value_counts().head(5))
print(ecom['Email'].apply(lambda email: email.split('@')[1]).value_counts().iloc[0:5])
print('------------------------------------------\n------------------------------------------')
