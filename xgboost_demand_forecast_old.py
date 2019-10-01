# -*- coding: 
#utf-8 -*-
"""
Created on Mon Apr 15 11:09:36 2019

@author: PC10
"""

from func_get_product_history import get_product_history
from class_products_db_finder import products_db_finder
import pickle
import pandas as pd
import os
from func_tratamento import tratamento
import datetime
import math
from sklearn.model_selection import PredefinedSplit
import tqdm
# finding product id
path,dic_name = r'C:\ProductClustering\productsDB\products_db_objects\\','products_db_dict'
g = os.path.join(os.path.dirname(path), dic_name)
prod_db = pd.DataFrame(pickle.load(open(g, 'rb')))


a = products_db_finder()
a.init_products_db(prod_db)
view = a.get_similar_products(title = 'pipoqueira  eletrica')

view[list(view.keys())[0]][['product_id','ad_title']]

# getting product historical data
history = get_product_history(product_id = [169073,19959], min_price = 500, max_price = 2500,drop_blackout = False)
history['date'] = pd.to_datetime(history['date'], errors  = 'coerce',format = '%Y-%m-%d')
#history = history[(history.daily_sales > min(history.daily_sales))&(history.daily_sales < max(history.daily_sales))]


len_= len(history)
daily_views = []
for (date, ad) in tqdm.tqdm(history[['date','ad_id']].values):
    try: 
        daily_views.append(max(history[(history.date== date)&(history.ad_id ==ad)].views.max()-history[(history.date== (date  - datetime.timedelta(1)))&(history.ad_id ==ad)].views.max(),0)) 
    except KeyError:
        daily_views.append(np.nan)
        
history = history.assign(daily_views = daily_views)


#agg function:
from func_gini import gini
import numpy as np
import datetime
def fu1(x):
        d = {}
        try:
            d['amount_of_ads']  = x['amount_of_ads'].sum()
        except:
            try:    
                d['amount_of_ads'] = x['date'].count()
            except:
                pass                    
        try:
            d['amount_of_sellers'] = x['seller_id'].nunique()
        except:
            d['amount_of_sellers'] = x['amount_of_sellers'].mean()
        d['daily_revenues'] = x['daily_revenues'].sum()        
        d['daily_sales'] = x['daily_sales'].sum()
        try:
            gini_coeff = gini(x.groupby('seller_id')['daily_sales'].sum())
        except:
            gini_coeff = x['gini'].mean()
        d['gini'] = gini_coeff if d['daily_sales'] >= 0 else np.nan
        try:
            d['min_price'] = x['price'].min()
            d['median_price'] =  x['price'].median()
            d['max_price'] = x['price'].max()
        except:
            d['min_price'] = x['min_price'].min()
            d['median_price'] =  x['median_price'].median()
            d['max_price'] = x['max_price'].max()
        try:
            d['month']  = x['date'].max().month
            d['day'] = x['date'].max().day
        except:
            pass
        try:
            d['daily_views'] = x['daily_views'].sum()
            d['daily_views_per_seller'] = x['daily_views'].sum()/d['amount_of_sellers']
        except:
            pass
        
        try:
            d['ad_type_1']  = x['ad_type_id'][x['ad_type_id'] == 1].count()/x['ad_type_id'].count()
            d['ad_type_3'] = x['ad_type_id'][x['ad_type_id'] == 3].count()/x['ad_type_id'].count()
        except:
            d['ad_type_1'] = x['ad_type_1'].mean()
            d['ad_type_3'] = x['ad_type_3'].mean()
        try:
            d['expected_demand'] = x['expected_demand'].sum()
        except KeyError:
            pass
        result = pd.Series(data = d)
        return result

#########################
period_after = 1
comparing_period = period_after
period_before_1 = 360 - period_after
period_before_2 = 30
assert comparing_period - period_after >= 0
assert period_before_1  > period_before_2 
avalible_features = ['amount_of_ads','date', 'amount_of_sellers', 'daily_revenues', 'daily_sales', 'gini','min_price', 'median_price', 'max_price', 'month', 'day', 'daily_views','daily_views_per_seller', 'ad_type_1', 'ad_type_3', 'expected_demand','week', 'year', 'quarter', 'prev_daily_sales']

prev_features_1 = ['daily_sales']
prev_features_2 =  ['daily_sales']
features = ['month','daily_sales','amount_of_sellers']
features += ['prev_1_{}'.format(i) for i in prev_features_1] + ['prev_2_{}'.format(i) for i in prev_features_2]

drop_na = False
train_test_split_period = period_after + comparing_period
resample_period = 'D'

###  plot
group_period = 'M'
#########################
from scipy import stats


history['date'] = pd.to_datetime(history['date'], errors  = 'coerce',format = '%Y-%m-%d')
groupped_data = history.groupby('date').apply(fu1).reset_index(drop=False)

#groupped_data.loc[groupped_data[groupped_data.date.isin(['2018-07-26','2018-11-11'])].index,:] = groupped_data.loc[groupped_data[groupped_data.date.isin(['2018-07-26','2018-11-11'])].index,:].assign(daily_sales = len(groupped_data.loc[groupped_data[groupped_data.date.isin(['2018-07-26','2018-11-11'])].index,:])*[0])
groupped_data[groupped_data.daily_sales < 0] = groupped_data[groupped_data.daily_sales < 0].assign(daily_sales = 0)
groupped_data = groupped_data[(stats.zscore(groupped_data['daily_sales']) < 3)]
groupped_data = groupped_data.resample(resample_period, on = 'date').apply(fu1).reset_index(drop =  False)

lista =[]
for date_i in groupped_data.date:
    try:        
        lista.append(groupped_data[groupped_data.date.between(date_i+datetime.timedelta(int(period_after-min(0.1*period_after,3))),date_i + datetime.timedelta(period_after+min(0.1*period_after,3)))].daily_sales.mean())
    except:
        lista.append(np.nan)

groupped_data = groupped_data.assign(expected_demand = np.array(lista))


week=[]
year = []
for i in range(len(groupped_data)):
    try:
        week.append(groupped_data.date[i].isocalendar()[1])
        year.append(groupped_data.date[i].isocalendar()[0])
    except:
        pass
groupped_data = groupped_data.assign(week = week,year= year) 

trimester = []
for m in groupped_data.month.values:
    trimester.append((m-1)//3 + 1)

groupped_data = groupped_data.assign(quarter = trimester)

####  model
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from matplotlib import pyplot

for feature in prev_features_1:
    previous =[]
    for date_i in groupped_data['date']:
        previous.append(groupped_data[groupped_data.date.between(date_i - datetime.timedelta(period_before_1*1.1),date_i - datetime.timedelta(period_before_1*0.9))][feature].mean())
    
    groupped_data = eval("groupped_data.assign({} = previous)".format('prev_1_{}'.format(feature)))

for feature in prev_features_2:
    previous =[]
    for date_i in groupped_data['date']:
        previous.append(groupped_data[groupped_data.date.between(date_i - datetime.timedelta(math.ceil(period_before_2*1.1)),date_i - datetime.timedelta(math.ceil(period_before_2*0.9)))][feature].mean())
    
    groupped_data = eval("groupped_data.assign({} = previous)".format('prev_2_{}'.format(feature)))




groupped_data_w_na = groupped_data


if drop_na:
    groupped_data = groupped_data.dropna()
else:
    groupped_data= groupped_data.fillna(method = 'backfill')


X_train_df = groupped_data[features+['date']][groupped_data.date <= groupped_data.date.max()- datetime.timedelta(train_test_split_period)]
cutting_date = X_train_df.date.max()
X_train_df = groupped_data[features][groupped_data.date <= groupped_data.date.max()- datetime.timedelta(train_test_split_period)]
X_test_df = groupped_data[features][(groupped_data.date > cutting_date)&(groupped_data.date <= cutting_date + datetime.timedelta(comparing_period))]
y_train_df, y_test_df = groupped_data[groupped_data.date <= groupped_data.date.max()- datetime.timedelta(train_test_split_period)][['expected_demand']], groupped_data[['expected_demand']][(groupped_data.date > cutting_date)&(groupped_data.date <= cutting_date + datetime.timedelta(comparing_period))]


X_train, X_test, y_train, y_test = X_train_df.values, X_test_df.values, y_train_df.values, y_test_df.values 

parameters = [{
        'colsample_bytree':[1],
        'max_depth':list(np.arange(1,X_train.shape[1],1)),
        'n_estimators':[1000],
        'early_stopping_rounds':[10],
        'booster':['gbtree'],
        'verbosity':[1],
        'subsample': list(np.linspace(0.25,1,4)),
        #'learning_rate':[0.0001,0.001,0.01,0.1],
        #'eval_set': [[(X_test,y_test)]],
        'gamma': [0,0.0001,0.001,0.01,0.1],
        'eval_metric': ['rmse'],
        'verbose' :[True],
        'silent' : [False],
        'min_child_weight':list(np.arange(1,X_test.shape[1],5)),
        'n_estimators': [10,100,200,300,1000]
        }]
        

xg_reg = xgb.XGBRegressor()

# Applying Grid Search to find the best model and the best parameters
from hypopt import GridSearch
from sklearn.model_selection import GridSearchCV
grid_search = GridSearch(model = xg_reg,
                           param_grid = parameters
                           )

grid_search = grid_search.fit(X_train, y_train,X_val = X_test,y_val = y_test,scoring = 'neg_mean_squared_error')
best_parameters = grid_search.get_params()


#best_mse = (-grid_search.best_score_)**(1/2)
#best_parameters = grid_search.best_params_


'''best_parameters['n_estimators'] = 10000
best_parameters['gamma'] = 0
best_parameters['min_child_weight'] = 2'''

xg_reg = xgb.XGBRegressor(**best_parameters)

xg_reg.fit(X_train,y_train)
    
preds = xg_reg.predict(X_test)

#preds = std_scaler_y.inverse_transform(preds)
#y_test = std_scaler_y.inverse_transform(np.nan_to_num(y_test.reshape(1,-1)[0]))
rmse = np.sqrt(mean_squared_error(np.nan_to_num(y_test), preds))/np.nanmean(y_test)
avg = np.sqrt(mean_squared_error(np.nan_to_num(y_test), len(y_test)*[np.mean(np.append(y_test,y_train))]))/np.nanmean(y_test)
median = np.sqrt(mean_squared_error(np.nan_to_num(y_test), len(y_test)*[np.median(np.append(y_test,y_train))]))/np.nanmean(y_test)
avg_recent = np.sqrt(mean_squared_error(np.nan_to_num(y_test), len(y_test)*[np.mean(y_train[-period_before//7:])]))/np.nanmean(y_test)


print("RMSE: %f" % (rmse))


########################

X_train_pred_df_plain, X_test_pred_df_plain = groupped_data[groupped_data.date <= groupped_data.date.max()- datetime.timedelta(train_test_split_period)].assign(pred = xg_reg.predict(X_train)), groupped_data[(groupped_data.date > cutting_date)&(groupped_data.date <= cutting_date + datetime.timedelta(comparing_period))].assign(pred = xg_reg.predict(X_test))
groupped_data_w_na['date'] = pd.to_datetime(groupped_data_w_na['date'], errors  = 'coerce',format = '%Y-%m-%d')


X_train_pred_df =  X_train_pred_df_plain.resample(group_period, on = 'date').mean().reset_index(drop = False)
X_test_pred_df = X_test_pred_df_plain.resample(group_period, on = 'date').mean().reset_index(drop = False)

pyplot.clf()
pyplot.plot(groupped_data_w_na.resample(group_period, on = 'date').date.max(),groupped_data_w_na.resample(group_period, on = 'date').daily_sales.mean(), color = 'g')
pyplot.plot(X_test_pred_df.date + datetime.timedelta(period_after) ,X_test_pred_df.pred , color = 'r')
pyplot.plot(X_train_pred_df.date + datetime.timedelta(period_after) ,X_train_pred_df.pred , color = 'b')


##################
pyplot.clf()
residuals = y_test.ravel()-preds
pyplot.scatter(y_test,residuals/y_test.ravel())
pyplot.show()
##plots
import pylab as plt
plt.clf()
plt.bar(range(len(xg_reg.feature_importances_)), xg_reg.feature_importances_)
plt.xticks(range(len(xg_reg.feature_importances_)),features)
plt.show()

pyplot.clf()
pyplot.scatter(groupped_data.prev_daily_sales,groupped_data.expected_demand)

corre = groupped_data.corr()


