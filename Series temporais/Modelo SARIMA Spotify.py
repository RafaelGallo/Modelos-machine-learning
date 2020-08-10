#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("Dados 2\spotify.csv")
df.head()


# In[4]:


df.drop("Despacito", inplace = True, axis = 1)


# In[5]:


df.drop("Something Just Like This", inplace = True, axis = 1)


# In[6]:


df.drop("HUMBLE.", inplace = True, axis = 1)


# In[7]:


df.drop("Unforgettable", inplace = True, axis = 1)


# In[8]:


df.head()


# In[9]:


df.isnull()


# In[12]:


df.columns = ["Data", "Pessoas"]
df.head()


# In[11]:


df.tail()


# In[13]:


df.describe()


# In[14]:


df.shape


# In[15]:


plt.figure(figsize=(20, 10))
plt.style.use('seaborn-darkgrid')

x1 = sns.lineplot(x="Data", y="Pessoas", data=df)
plt.xticks(rotation=70);
plt.show()


# In[16]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df["Pessoas"]);


# In[17]:


plot_pacf(df["Pessoas"]);


# **Previsoes**

# In[18]:


df_x1 = df[:26][:]
df_x2 = df[26:][:]


# **Modelo SARIMA**

# In[19]:


from pmdarima.arima.utils import nsdiffs

D = nsdiffs(df['Pessoas'].values, m=2, max_D=12, test='ch')
D


# In[31]:


from pmdarima.arima import auto_arima

modelo_sarima = auto_arima(df['Pessoas'].values,start_p = 0, start_q = 0, max_p = 6, max_q = 6, d = 1,D = 1,
                            start_Q = 2, start_P = 2, max_P = 5, max_Q = 5, m = 3, seasonal = True, trace = True, 
                            error_action ='ignore',
                            suppress_warnings = True, stepwise = False, maxiter = 50)

modelo_sarima.aic()


# In[32]:


modelo_sarima.fit(df_x1["Pessoas"]. values)

ml_predict = modelo_sarima.predict(n_periods = 10)
ml_predict


# In[33]:


plt.figure(figsize = (35, 25))

plt.plot(ml_predict)
plt.plot(df["Data"], df["Pessoas"])


# In[ ]:




