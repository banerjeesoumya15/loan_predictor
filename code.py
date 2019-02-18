
# coding: utf-8

# In[2]:


'''
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
'''

import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv("D:\\DataScience\\loan_predictor\\train.csv")


# In[7]:


train.head(10)


# In[8]:


#Summarise numerical variables
train.describe()


# In[9]:


#For non numerical variable
train['Property_Area'].value_counts()


# In[14]:


'''Distribution Analysis'''
#Histogram
train['ApplicantIncome'].hist(bins=50, color='blue')


# In[15]:


#to display outliers
train.boxplot(column='ApplicantIncome')


# In[16]:


'''This confirms the presence of a lot of outliers/extreme values. 
This can be attributed to the income disparity in the society. 
Part of this can be driven by the fact that we are looking at people with different education levels. 
Let us segregate them by Education:'''
train.boxplot(column='ApplicantIncome', by='Education')


# In[17]:


#histogram of loan amount
train['LoanAmount'].hist(bins=50)


# In[18]:


#boxplot of Loan Amount
train.boxplot(column='LoanAmount')


# In[4]:


'''Categorical variable analysis'''
temp1 = train['Credit_History'].value_counts(ascending = True)
temp2 = train.pivot_table(values = 'Loan_Status', index = ['Credit_History'], aggfunc = lambda x: x.map({'Y':1, 'N':0}).mean())
print('Frequency table for Credit History:')
print(temp1)

print('\nProbability of getting a loanfor each Credit History class:')
print(temp2)


# In[8]:


# matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title('Applicants by credit history')
temp1.plot(kind = 'bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title('Probability of getting loan by credit history')

# Similar graphs can be created for other variables like 
# Married, Self_Employed, Property_Area


# In[9]:


# crosstab
temp3 = pd.crosstab(train['Credit_History'], train['Loan_Status'])
print(temp3)


# In[10]:


temp3.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)


# In[20]:


temp4 = pd.crosstab(index = [train['Credit_History'], train['Gender']], columns = train['Loan_Status'])
print(temp4)
temp4.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)


# In[23]:


'''Data munging'''
# Check missing values
train.apply(lambda x: sum(x.isnull()), axis = 0)  # by default axis = 0
# we even have to check for unpractical values for variables
# e.g. Loan Amount = 0 is unpractical


# In[29]:


# fill missing Loan Amount - Several ways are there
# One way
# train['Loan_Amount'].fillna(train['Loan_Amount'].mean(), inplace = True)

# Another way
# to fill loan amount as per median values of self employment and education
# First, let’s look at the boxplot to see if a trend exists:
train.boxplot(column = 'LoanAmount', by = ['Education', 'Self_Employed'])


# In[31]:


# we will use this median values of loan amount for each group to impute the values
# first we ensure Education and Self_Employed should not have any missing values
train['Self_Employed'].value_counts()


# In[32]:


# ~86% are No. So we impute No to missing values
train['Self_Employed'].fillna('No', inplace = True)


# In[36]:


# we create pivot table which provides median values for each unique combination of Education and Self_Employed
table = train.pivot_table(values = 'LoanAmount', index = 'Self_Employed', columns = 'Education', aggfunc = np.median)
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]
train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis = 1), inplace = True)


# In[ ]:


'''Treat extreme values in LoanAmount and ApplicantIncome'''

