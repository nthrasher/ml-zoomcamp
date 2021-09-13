#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd
import numpy as np
from numpy.linalg import inv


# In[126]:


# Answer 1
print(f"Answer 1: {np.__version__}")


# In[127]:


# Answer 2
print(f"Answer 2: {pd.__version__}")


# In[128]:


df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv")
df.head()


# In[150]:


# Answer 3
bmw_avg = df[df["Make"] == "BMW"]["MSRP"].mean().round(2)
print(f"Answer 3: {bmw_avg}")


# In[132]:


fifteen_or_after_df = df[df["Year"] >= 2015]
fifteen_or_after_df.head()


# In[133]:


# Answer 4
print(f"Answer 4: {fifteen_or_after_df['Engine HP'].isnull().sum()}")


# In[134]:


hp_before_avg = fifteen_or_after_df["Engine HP"].mean()
hp_before_avg


# In[135]:


fifteen_or_after_df = fifteen_or_after_df.fillna(value={"Engine HP": hp_before_avg})
fifteen_or_after_df.head()


# In[136]:


# Check my work
fifteen_or_after_df["Engine HP"].isnull().sum()


# In[137]:


hp_after_avg = fifteen_or_after_df["Engine HP"].mean()
hp_after_avg


# In[138]:


# Answer 5 - No Change
print("Answer 5:")
print(round(hp_before_avg))
print(round(hp_after_avg))


# In[139]:


rolls_royce_df = df[df["Make"] == "Rolls-Royce"]
rolls_royce_df.head()


# In[140]:


rolls_royce_df = rolls_royce_df[["Engine HP", "Engine Cylinders", "highway MPG"]]
rolls_royce_df.head()


# In[141]:


# Per the homework, should only be 7 rows
rolls_royce_df = rolls_royce_df.drop_duplicates()
rolls_royce_df


# In[142]:


X = rolls_royce_df.values
X


# In[143]:


X_t = X.T
X_t


# In[144]:


XTX = X_t.dot(X)
XTX_inv = inv(XTX)
XTX_inv


# In[145]:


# Answer 6
print(f"Answer 6: {XTX_inv.sum()}")


# In[146]:


y = [1000, 1100, 900, 1200, 1000, 850, 1300]
y


# In[147]:


X_t_times_XTX_inv = XTX_inv.dot(X.T)
X_t_times_XTX_inv


# In[151]:


# Answer 7
w = X_t_times_XTX_inv.dot(y)
print(f"Answer 7: {w[0]}")

