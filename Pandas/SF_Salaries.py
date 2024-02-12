'''
SF Salaries Exercise 
'''

# Import pandas as pd
import pandas as pd

# Read Salaries.csv as a dataframe called sal
sal = pd.read_csv("Pandas\\Salaries.csv")

# Check the head of the DataFrame
print(sal.head())
print('------------------------------------------\n------------------------------------------')

# Use the .info() method to find out how many entries there are
print(sal.info())
print('------------------------------------------\n------------------------------------------')

# What is the average BasePay?
print(sal['BasePay'].mean())
print('------------------------------------------\n------------------------------------------')

# What is the highest amount of OvertimePay in the dataset?
print(sal['OvertimePay'].max())
print('------------------------------------------\n------------------------------------------')

# What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll)
print(sal['JobTitle'].loc[sal['EmployeeName'] == 'JOSEPH DRISCOLL'])
print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle'])
print('------------------------------------------\n------------------------------------------')

# How much does JOSEPH DRISCOLL make (including benefits)?
pay = sal['TotalPayBenefits'].loc[sal['EmployeeName'] == 'JOSEPH DRISCOLL'].iloc[0]
print(pay)
print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits'].iloc[0])
print('------------------------------------------\n------------------------------------------')

# What is the name of highest paid person (including benefits)?
print(sal['EmployeeName'].loc[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()])
print(sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName'])
print(sal.loc[sal['TotalPayBenefits'].idxmax()]['EmployeeName'])
print(sal.iloc[sal['TotalPayBenefits'].argmax()])
print('------------------------------------------\n------------------------------------------')

# What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?
print(sal.loc[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()])
print('------------------------------------------\n------------------------------------------')

# What was the average (mean) BasePay of all employees per year? (2011-2014)?
print(f'2011 - {sal['BasePay'].loc[sal['Year'] == 2011].mean()}')
print(f'2012 - {sal['BasePay'].loc[sal['Year'] == 2012].mean()}')
print(f'2013 - {sal['BasePay'].loc[sal['Year'] == 2013].mean()}')
print(f'2014 - {sal['BasePay'].loc[sal['Year'] == 2014].mean()}')
print(sal[['Year', 'BasePay']].groupby('Year').mean())
print('------------------------------------------\n------------------------------------------')

# How many unique job titles are there?
print(sal['JobTitle'].nunique())
print('------------------------------------------\n------------------------------------------')

# What are the top 5 most common jobs?
def common_jobs(series : pd.Series):
    count = {}
    for val in series:
        if val not in count.keys():
            count[val] = 1
        else:
            count[val] += 1
            
    maxs = {}
    while len(maxs.keys()) < 6:
        max_value = ['', 0]
        for key in count.keys():
            if key not in  maxs.keys():
                if count[key] > max_value[1]:
                    max_value = [key, count[key]]
        maxs[max_value[0]] = max_value[1]
    return maxs
        
print(common_jobs(sal['JobTitle']))
print(sal['JobTitle'].value_counts().iloc[0:5])
print(sal['JobTitle'].value_counts().head(5))
print('------------------------------------------\n------------------------------------------')

# How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)
print(sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1))
print('------------------------------------------\n------------------------------------------')

# How many people have the word Chief in their job title?
def has_chief(series: pd.Series):
    chiefs = 0
    for val in series:
        if 'chief' in val.lower():
            chiefs += 1
    return chiefs

print(has_chief(sal['JobTitle']))
print(sal['JobTitle'][sal['JobTitle'].str.contains('chief', case = False)].count())
print('------------------------------------------\n------------------------------------------')

# Bonus: Is there a correlation between length of the Job Title string and Salary?
sal['job_title_len'] = sal['JobTitle'].apply(len)
print(sal[['job_title_len', 'TotalPayBenefits']].corr())
print('------------------------------------------\n------------------------------------------')
