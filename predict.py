#!/usr/bin/env python
# coding: utf-8

# ## ARIMA Time Series Modeling for XPRIZE competition

# In[32]:


import numpy as np
import pandas as pd
import scipy
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


import warnings
warnings.filterwarnings('ignore')
import glob

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


base = dt.datetime.today() - dt.timedelta(days=1)
numdays = ( dt.datetime.today() - dt.datetime(2020,1,22)).days
date_list = [ (base - dt.timedelta(days=x)).strftime("%m-%d-%Y") for x in range(numdays)]
#print(date_list)


# In[3]:


import glob

# get data file names
path =r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'

dfs = []
for date in date_list:    
    fullpath = path + date + ".csv"
    #print (fullpath)
    dfs.append(pd.read_csv(fullpath))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)    


# In[4]:


big_frame.head()


# In[5]:


big_frame.info()


# In[6]:


big_frame = big_frame[big_frame.Province_State != "Province_State"]


# In[7]:


big_frame['Last_Update'] = big_frame['Last_Update'].astype('datetime64[ns]').dt.date


# In[8]:


df_sum = big_frame.groupby(["Country_Region","Last_Update"],as_index=False).sum()
df_sum.head()


# In[9]:


df_sum = df_sum.assign(rec_id=np.arange(len(df_sum))).reset_index(drop=True)
df_sum.info()


# In[10]:


df_sum.drop(columns=['FIPS',"Lat","Long_","Incident_Rate", "Incidence_Rate","Case_Fatality_Ratio", "Case-Fatality_Ratio","Latitude","Longitude",],inplace=True)


# In[11]:


Country_Region = df_sum.Country_Region.str.strip()
df_sum.update(pd.DataFrame(Country_Region,columns=["Country_Region"]))


# In[12]:


C1 = df_sum.Country_Region.str.replace("*","")
C2 = C1.str.replace(")","")
C3 = C2.str.replace("(","")
df_sum.update(pd.DataFrame(C3,columns=["Country_Region"]))


# In[13]:


df_sum.sort_values(["Country_Region","Last_Update"],inplace=True)


# In[14]:


df_sum.rename(columns= {"Last_Update":"Dated", "Confirmed":"total_cases","Deaths":"total_deaths"},inplace=True)


# In[15]:


df_clean = df_sum.groupby(["Country_Region","Dated"],as_index=False).sum()


# In[16]:


df_clean = df_clean[df_clean.Country_Region.notnull()]


# In[17]:


df_clean.sort_values(["Dated"],inplace=True)


# In[18]:


df_clean.head()


# In[19]:


countries = pd.unique(df_clean.Country_Region)
#countries


# In[20]:


df_clean.set_index(pd.to_datetime(df_clean.Dated), inplace=True)
df_clean.drop(columns = ['Dated'],inplace=True)


# In[61]:



# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

def evaluate_models(dataset, p_values, d, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					#print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue   
	return best_cfg #print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# In[77]:


out = pd.DataFrame()

for i in ['China','Canada']: #loop over each countries data
    df = df_clean[df_clean['Country_Region']==i]
    df.sort_values(["Dated"],inplace=True)
    df = df.cumsum()    
    df.drop(columns = ['rec_id'],inplace=True)
    df = df.resample('D').ffill()
    df.Country_Region = i  
    
    df['new_cases'] = df.total_cases -  df.total_cases.shift()

    model1 = ARIMA(df.new_cases[1:], order= evaluate_models(df.new_cases[1:],[1,2,3],0,[1,2,3]))
    model1_fit = model1.fit()
    predictions = model1_fit.predict(start=pd.to_datetime("12-22-2020"),end=pd.to_datetime("2-1-2021"))

    finaldf = pd.DataFrame(predictions,columns = ['PredictedDailyNewCases'])
    finaldf.reset_index(inplace=True)
    finaldf.rename(columns = {"index":"Date"},inplace=True)
    finaldf['CountryName'] = i
    finaldf['RegionName'] = None
    finaldf = finaldf[['CountryName','RegionName','Date','PredictedDailyNewCases']]
    out = pd.concat([out,finaldf])


# In[78]:


out


# In[79]:


out.to_csv("predictions.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




