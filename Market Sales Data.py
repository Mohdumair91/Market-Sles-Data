#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("train.csv")
df.head()


# In[2]:


df.shape


# # Preprocessing

# In[3]:


df.isna().sum()


# In[4]:


df1=df


# In[5]:


## As we see that Item_weight and Outlet_size has some NULL values so Handle them


# In[6]:


## As Item_weight is continous values so replace them with the mean of that column
import numpy as np
df1["Item_Weight"].replace(np.nan,df1["Item_Weight"].mean(),inplace=True)


# In[7]:


df1["Item_Weight"].isna().sum()


# In[8]:


## As Outlet_size has Discreste value so replace them with the mode
print(df1["Outlet_Size"].mode())
df1["Outlet_Size"].replace(np.NaN,df1["Outlet_Size"].mode()[0],inplace=True)


# In[9]:


df1["Outlet_Size"].isna().sum()


# In[10]:


df1.head()


# # Exploratory Data Analysis

# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[12]:


## As we see Medium Outlet_size is maximum
sns.countplot(df1["Outlet_Size"])


# In[13]:


df2=df1.groupby("Item_Fat_Content")
df2.size()


# In[14]:


## As we see LF,Low Fat and low fat represent the same thing so represent them with same name


# In[15]:


df3=df1


# In[16]:


df3["Item_Fat_Content"]=df3["Item_Fat_Content"].replace("LF","Low Fat")


# In[17]:


df3["Item_Fat_Content"]=df3["Item_Fat_Content"].replace("low fat","Low Fat")
df2=df3.groupby("Item_Fat_Content")
df2.size()


# In[18]:


## Check mean


# In[19]:


df3.groupby("Outlet_Size").mean()


# In[20]:


k=df3.groupby("Outlet_Size").Item_Outlet_Sales.mean()
print(k.index)
print(k.values)


# In[21]:


sns.barplot(x=k.index,y=k.values)


# In[22]:


df3.head()


# In[23]:


df_items=df3.groupby("Item_Type").Item_Outlet_Sales.sum()
df_items


# In[24]:


# As we see that the sale of Fruits and Vegitables, snack Foods are maximum
plt.figure(figsize=(18,6))
sns.barplot(x=df_items.index,y=df_items.values)


# In[25]:


df_location=df3.groupby("Outlet_Location_Type").Item_Outlet_Sales.sum()
df_location


# In[26]:


# Tier 3 has maximum sales
sns.barplot(x=df_location.index,y=df_location.values)


# In[27]:


df3.head()


# In[28]:


# As we see that in 1985 maximum number of Outlet_Establishment 
sns.countplot(df3["Outlet_Establishment_Year"])


# In[29]:


# Super market type 1 has maximum sales
plt.figure(figsize=(14,5))
df_Outlet_Type=df3.groupby("Outlet_Type").Item_Outlet_Sales.sum()
sns.barplot(x=df_Outlet_Type.index,y=df_Outlet_Type.values)


# # One Hot Encoding of data

# In[30]:


df3.head()


# In[31]:


from sklearn.preprocessing import OneHotEncoder


# In[32]:


df3["Outlet_Location_Type"].unique()


# In[ ]:





# In[ ]:





# In[33]:


enc=OneHotEncoder()
enc_data=pd.DataFrame(enc.fit_transform(df3[['Item_Type','Outlet_Establishment_Year','Outlet_Type','Outlet_Location_Type','Outlet_Type']]).toarray())


# In[34]:


enc_data.head()


# In[35]:


df4=df3
df4.head()


# In[37]:


df4.drop(['Item_Identifier','Item_Fat_Content','Item_Type','Item_Visibility','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type'],axis=1,inplace=True)


# In[38]:


df4.head()


# In[39]:


df5=df4.join(enc_data)


# In[40]:


df5.head()


# In[41]:


y=df5["Item_Outlet_Sales"]
x=df5.drop(['Item_Outlet_Sales'],axis=1)


# In[42]:


x.head()


# In[43]:


y.head()


# In[ ]:





# # Model Building

# In[44]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[45]:


x_train.head()


# In[46]:


from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(x_train,y_train)


# In[47]:


y_pre=model1.predict(x_test)


# In[48]:


y_pre


# In[49]:


y_test


# In[50]:


model1.score(x_test,y_test)


# In[51]:


model1.fit(x,y)


# ## Test On Testing data

# In[52]:


df_test=pd.read_csv("test.csv")
df_test.head()


# In[53]:


df_test.isna().sum()


# In[54]:


df1_test=df_test


# In[55]:


## As Item_weight is continous values so replace them with the mean of that column
import numpy as np
df1_test["Item_Weight"].replace(np.nan,df1_test["Item_Weight"].mean(),inplace=True)


# In[57]:


## As Outlet_size has Discreste value so replace them with the mode
print(df1_test["Outlet_Size"].mode())
df1_test["Outlet_Size"].replace(np.NaN,df1_test["Outlet_Size"].mode()[0],inplace=True)


# In[58]:


enc=OneHotEncoder()
enc_data=pd.DataFrame(enc.fit_transform(df1_test[['Item_Type','Outlet_Establishment_Year','Outlet_Type','Outlet_Location_Type','Outlet_Type']]).toarray())


# In[59]:


df2_test=df1_test
df2_test.head()


# In[60]:


df2_test.drop(['Item_Identifier','Item_Fat_Content','Item_Type','Item_Visibility','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type'],axis=1,inplace=True)


# In[62]:


df3_test=df2_test.join(enc_data)


# In[63]:


df3_test.head()


# In[64]:


y_predicted=model1.predict(df3_test)


# In[65]:


y_predicted


# In[ ]:




