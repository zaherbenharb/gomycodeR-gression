#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


# 

# In[7]:


data =pd.read_csv("C:\\Users\\Zed\\Desktop\\gomycode/kc_house_data.csv")
data.head()


# In[10]:


plt.title("histogram of diffrent price")
plt.xlabel("price")
data['price'].plot.hist()


# In[11]:


sns.set_theme(color_codes=True)

ax = sns.regplot(x="price", y="yr_built", data=data)


# In[12]:


data = data[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','yr_built','zipcode','lat','long']]
data.head()


# In[ ]:





# In[13]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[14]:


plot_correlation_map(data)


# In[15]:


data.info()


# In[16]:


x = data['sqft_living'].values[:,np.newaxis]
y = data['price'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=30)


# In[17]:


model = LinearRegression()


# In[18]:


model.fit(x_train,y_train)
predicted=model.predict(x_test)


# In[19]:


print("MSE", mean_squared_error(y_test,predicted))
print("R squared", metrics.r2_score(y_test,predicted))


# In[20]:


plt.scatter(x,y,color="r")
plt.title("Linear Regression")
plt.ylabel("price")
plt.xlabel("sqft_living")
plt.plot(x,model.predict(x),color="k")
plt.show()


# In[28]:


x=data[["sqft_living","bathrooms","view","bedrooms","long","lat"]]  
y=data["price"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=40) 

model=LinearRegression() 
model.fit(x_train,y_train) 
predicted=model.predict(x_test) 

print("MSE", mean_squared_error(y_test,predicted))
print("R squared", metrics.r2_score(y_test,predicted))


# In[27]:


x=data[["sqft_living","view","bedrooms","sqft_lot","zipcode","long","lat"]]  
y=data["price"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=40)  
lg=LinearRegression()
poly=PolynomialFeatures(degree=2)

x_train_fit = poly.fit_transform(x_train) 
lg.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = lg.predict(x_test_)

print("MSE: ", metrics.mean_squared_error(y_test, predicted))
print("R squared: ", metrics.r2_score(y_test,predicted))


# In[ ]:





# In[ ]:




