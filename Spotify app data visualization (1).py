#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('C:\\Users\\SANGHAMITRA KAR\\Downloads\\archive (3)\\tracks.csv')


# In[3]:


df.head()


# In[4]:


# null values

pd.isnull(df).sum()


# In[5]:


df.info()


# In[6]:


sorted_df = df.sort_values('popularity',ascending = True).head(10)
sorted_df


# In[7]:


df.describe().transpose()


# In[8]:


most_popular=df.query('popularity>90',inplace = False).sort_values('popularity', ascending=False).head(10)
most_popular


# In[9]:


df.set_index("release_date", inplace=True)
df.index=pd.to_datetime(df.index)
df.head()


# In[10]:


df[["artists"]].iloc[18]


# In[11]:


df["duration"]=df["duration_ms"].apply(lambda x: round(x/1000))
df.drop("duration_ms",inplace=True,axis=1)


# In[12]:


df.duration.head()


# In[13]:


corr_df=df.drop(["key","mode","explicit"],axis=1).corr(method="pearson")
plt.figure(figsize=(14,6))
heatmap=sns.heatmap(corr_df,annot=True,fmt=".1g",vmin=-1,center=0,cmap="inferno",linewidths=1,linecolor="Black")
heatmap.set_title("correlation Heatmap Between Variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)


# In[14]:


sample_df = df.sample(int(0.004*len(df)))


# In[15]:


print(len(sample_df))


# In[16]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y = "loudness", x = "energy" , color = 'c').set(title="loudness vs Energy Correlation")


# In[17]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y = "popularity", x = "acousticness" , color = 'r').set(title="popularity vs acoustickness")


# In[18]:


df['dates']=df.index.get_level_values('release_date')
df.dates=pd.to_datetime(df.dates)
years=df.dates.dt.year


# In[19]:


#pip install ---


# In[20]:


sns.displot(years,discrete=True,aspect=2,height=5,kind="hist").set(title="Number of songs per year")


# In[21]:


total_dr = df.duration
fig_dims = (18,7)
fig, ax =plt.subplots(figsize = fig_dims)
fig = sns.barplot(x=years, y = total_dr, ax = ax, errwidth = False).set(title="Year vs Duration")
plt.xticks(rotation=90)


# In[ ]:




