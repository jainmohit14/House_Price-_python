#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


data = pd.read_csv('house_price_regression_dataset.csv')
print(data.head())


# In[19]:


data.columns


# In[21]:


data.shape


# In[33]:


missing =data.isnull().sum()
missing =missing[missing == 0]
missing.sort_values(inplace =True)
missing.plot.bar()
                    


# In[44]:


sns.set(rc={'figure.figsize':(12,8)})
sns.displot(data['House_Price'],kde=False,bins=20);


# In[46]:


sns.kdeplot(data['House_Price'])


# In[50]:


data['House_Price'].describe()


# In[52]:


numeric_features = data.select_dtypes(include =[np.number])
numeric_features.columns


# In[60]:


categorical_features =data.select_dtypes(include=[object])
categorical_features.columns


# In[62]:


correlation = numeric_features.corr()
print(correlation['House_Price'].sort_values(ascending = False),'\n')
      


# In[64]:


f , ax=plt.subplots(figsize =(14,12))
plt.title('Correlation of Numeric Features with House_Price',y=1,size=16)
sns.heatmap(correlation,square = True, vmax=0.8)


# In[68]:


k=11
cols =correlation.nlargest(k,'House_Price')['House_Price'].index
print(cols)
cm=np.corrcoef(data[cols].values.T)
f , ax=plt.subplots(figsize =(14,12))
sns.heatmap(cm, vmax=8,linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values,annot_kws ={'size':12},yticklabels =cols.values)


# In[72]:


sns.scatterplot(x='Garage_Size',y='House_Price',data = data)


# In[74]:


sns.boxplot(x=data["House_Price"])


# In[102]:


f, ax = plt.subplots(figsize=(16,10))
fig =sns.boxplot(x='Year_Built',y='House_Price',data=data)
fig.axis(ymin=0,ymax=8000000);
xt =plt.xticks(rotation=45)


# In[108]:


first_quartile =data['House_Price'].quantile(.25)
third_quartile =data['House_Price'].quantile(.75)
IQR = third_quartile-first_quartile


# In[110]:


new_boundary = third_quartile + 3*IQR


# In[ ]:





# In[ ]:





# In[ ]:




