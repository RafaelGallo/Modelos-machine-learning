#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Dados 2\spotify.csv")
df.head()


# In[3]:


df.drop("Despacito", inplace = True, axis = 1)


# In[5]:


df.drop("Something Just Like This", inplace = True, axis = 1)


# In[6]:


df.drop("HUMBLE.", inplace = True, axis = 1)


# In[7]:


df.drop("Unforgettable", inplace = True, axis = 1)


# In[10]:


df.head()


# In[39]:


df.isnull()


# In[11]:


df.columns = ["Data", "Pessoas"]
df


# In[13]:


df.tail()


# In[14]:


df.describe()


# In[12]:


df.shape


# In[32]:


plt.figure(figsize=(20, 10))
plt.style.use('seaborn-darkgrid')

x1 = sns.lineplot(x="Data", y="Pessoas", data=df)
plt.xticks(rotation=70);
plt.show()


# In[33]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df["Pessoas"]);


# In[34]:


plot_pacf(df["Pessoas"]);


# **Previsoes**

# In[35]:


df_x1 = df[:26][:]
df_x2 = df[26:][:]


# **Modelo ARIMA**

# In[19]:


from pmdarima.arima import auto_arima

modelo_arima = auto_arima(df_x1["Pessoas"].values, start_p = 0, start_q = 0,
                         max_p = 8, max_q = 8, d = 2, seasonal = False, trace = True,
                         error_action = "ignore", suppress_warnings = True, stepwise = False)


# In[21]:


modelo_arima.aic()


# In[36]:


modelo_arima.fit(df_x1["Pessoas"]. values)

model_predict = modelo_arima.predict(n_periods = 10)
model_predict


# In[46]:


plt.figure(figsize=(30,25))

plt.plot(df["Data"], df["Pessoas"])
plt.plot(model_predict)


# In[ ]:




