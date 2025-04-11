#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas')


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("WildBunch_timestamps.csv")
df_ground_truth = pd.read_csv("wild_bunch_2.csv")


# In[3]:


df


# In[4]:


df_ground_truth["Start"] = pd.to_timedelta(df_ground_truth["Start"]).dt.seconds
df_ground_truth["End"] = pd.to_timedelta(df_ground_truth["End"]).dt.seconds


# In[5]:


df_ground_truth


# In[6]:


df = df[:len(df_ground_truth)]


# In[7]:


df["start_time"] = df["start_time"].apply(lambda x : x.replace("sec",""))
df["end_time"] = df["end_time"].apply(lambda x : x.replace("sec",""))


# In[8]:


df["start_time"] = df["start_time"].astype('float64')
df["end_time"] = df["end_time"].astype('float64')


# In[9]:


df.info()


# In[10]:


merged_df = pd.concat([df_ground_truth,df],axis=1)


# In[11]:


merged_df


# In[12]:


merged_df["Start Error"] = (merged_df["Start"] - merged_df["start_time"]).abs()
merged_df["End Error"] = (merged_df["End"] - merged_df["end_time"]).abs()


# In[13]:


merged_df


# In[14]:


merged_df.describe()


# In[15]:


merged_df[["Start Error","End Error"]].plot(kind="line")
plt.savefig("analysis.png")


# In[16]:


merged_df[["Start Error","End Error"]].plot(kind="box")
plt.savefig("analysis1.png")


# In[ ]:





# In[ ]:





# In[ ]:




