# -*- coding: 
#utf-8 -*-
"""
Created on Mon Apr 15 11:09:36 2019

@author: PC10
"""
import numpy as np
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
import TimeSeriesUtils as TSU
from matplotlib import pyplot as plt
import xgboost as xgb
from hypopt import GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
['date','amount_of_ads','active_seller','category_id','daily_sales_sum',
 'ad_type_mean','daily_views_sum','price_median','position_median',
 'sold_quantity_sum','gini_ads','conversion','share']

features = ['market_size','price_min','position_max','amount_of_ads','daily_views_sum','price_median','position_median','sold_quantity_sum','gini_ads','conversion','daily_sales_sum']
one_hot_features = []
pred_period = 50
look_back_period = 30
time_blocks = 3
foward_pred_goal = 15
seller_val= None

assert foward_pred_goal <= pred_period

X, sellers, features,views,X_dates = TSU.get_and_prepare_product_data(
        min_sold = 30,
        product_id = [5870],
        min_price = 100,
        max_price = 800,
        drop_blackout = False,
        features = features,
        one_hot_features = one_hot_features,
        dependent_variable = ['daily_sales_sum'])

if seller_val:
    seller_val = sellers.index(seller_val)

# getting product historical data

#
sellers
plt.clf()
view_seller.fillna('mean').rolling('{}d'.format(pred_period)).mean().tshift(periods = 0, freq = 'D').daily_sales_sum.plot()
view_seller.fillna('mean').tshift(periods = 2*pred_period, freq = 'D').rolling('{}d'.format(pred_period)).mean().daily_sales_sum.plot()


moving_average_error = {}
for seller  in sellers:
    view_seller = complete_view.loc[(slice(None),seller),:].reset_index(level = 1).fillna(0)
    moving_average_error[seller] =(abs(view_seller.fillna('mean').rolling('{}d'.format(pred_period)).mean().tshift(periods = 0, freq = 'D').daily_sales_sum-view_seller.fillna('mean').tshift(periods = 2*pred_period, freq = 'D').rolling('{}d'.format(pred_period)).mean().daily_sales_sum).mean())


#


X_dynamics = TSU.df_to_array(complete_view.fillna(0)[dynamic_vars])
X_static = TSU.df_to_array(complete_view.fillna(0)[static_vars])
X_seller_invar = TSU.df_to_array(complete_view.fillna(0)[seller_invariant_dynamic_vars].reset_index(level = 1).groupby(level = 0).max().reset_index().set_index(['date','seller_id']))
y_dynamic = TSU.df_to_array(complete_view.fillna(0)[dynamic_vars])

X_t_list=[]
y_t_list=[]
X_v_list=[]
y_v_list=[]

for seller in sellers:
    
    seller_pos = sellers.index(seller)
    X_train_d, y_train_i , X_val_d, y_val_i = TSU.chunk_to_pooled_2d(X_dynamics,pred_period,look_back_period,seller_pos ,time_blocks = time_blocks, functions = [np.mean, func_slope],flatten = False)
    X_train_si, _, X_val_si, _= TSU.chunk_to_pooled_2d(X_seller_invar,pred_period,look_back_period,seller_pos ,time_blocks = time_blocks, functions = [np.mean, func_slope],  flatten = False)
    X_train_st, _, X_val_st, _ = TSU.chunk_data_by_date(X_static,pred_period,look_back_period, static = True)
    _,y_train_i,_,y_val_i = TSU.chunk_data_by_date(y_dynamic,pred_period,look_back_period, flatten = True)
    y_train_i  = y_train_i[:,:,seller_pos].mean(axis = 1)
    y_val_i  = y_val_i[:,:,seller_pos].mean(axis = 1)
    seller_X_train = np.concatenate((np.take(X_train_st,[seller_pos],axis = -2),np.take(X_train_d,[seller_pos],axis = -2)),axis = -1)
    seller_X_val = np.concatenate((np.take(X_val_st,[seller_pos],axis = -2),np.take(X_val_d,[seller_pos],axis = -2)),axis = -1)
    
    seller_vars = static_vars + ['look_back_mean'+ var for var in dynamic_vars] + [ 'look_back_slope'+ var for var in dynamic_vars] 
    
    seller_X_train = seller_X_train.reshape(seller_X_train.shape[0],seller_X_train.shape[1]*seller_X_train.shape[2])
    seller_X_val = seller_X_val.reshape(seller_X_val.shape[0],seller_X_val.shape[1]*seller_X_val.shape[2])
    
    X_train_d = X_train_d.reshape(X_train_d.shape[0],X_train_d.shape[1]*X_train_d.shape[2])
    X_val_d = X_val_d.reshape(X_val_d.shape[0],X_val_d.shape[1]*X_val_d.shape[2])
    X_train_si = X_train_si.reshape(X_train_si.shape[0],X_train_si.shape[1]*X_train_si.shape[2])
    X_val_si = X_val_si.reshape(X_val_si.shape[0],X_val_si.shape[1]*X_val_si.shape[2])
    X_train_st = X_train_st.reshape(X_train_st.shape[0],X_train_st.shape[1]*X_train_st.shape[2])
    X_val_st = X_val_st.reshape(X_val_st.shape[0],X_val_st.shape[1]*X_val_st.shape[2])
    
    X_train_i = np.concatenate((seller_X_train,X_train_st,X_train_d,X_train_si),axis = 1)
    X_val_i = np.concatenate((seller_X_val,X_val_st,X_val_d,X_val_si),axis = 1)
    
    if seller_pos == 0:
        X_train = X_train_i
        X_val =  X_val_i
        y_train = y_train_i
        y_val = y_val_i
    else:
        X_train = np.concatenate((X_train,X_train_i),axis=0)
        X_val =  np.concatenate((X_val,X_val_i),axis=0)
        y_train =np.concatenate((y_train,y_train_i),axis=0)
        y_val = np.concatenate((y_val,y_val_i),axis=0)
        
    X_t_list.append(X_train_i)
    y_t_list.append(y_train_i)
    X_v_list.append(X_val_i)
    y_v_list.append(y_val_i)
    
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
#####

plt.clf()
for seller in range(len(y_t_list)):
    plt.plot(dates[look_back_period:-2*pred_period],y_t_list[seller])
    plt.plot(dates[-2*pred_period:-1*pred_period],y_v_list[seller])

X_train.shape

#agg function:

parameters = [{
        'min_child_weight':[1],
        'max_depth':[2,4,5],        
        'early_stopping_rounds':[10],
        'booster':['gbtree'],
        'verbosity':[1],
        'subsample': [0.75],
        #'learning_rate':[0.0001,0.001,0.01,0.1],
        'eval_set': [seller_val_set],
        'gamma': [0.1],
        'eval_metric': ['rmse'],
        'verbose' :[True],
        'silent' : [False],
        'min_child_weight': [1],
        'n_estimators': [100],
        'colsample_bytree': [0.1,0.2,0.3],
        'reg_lambda': [1],
        'reg_alpha': [0]
        }]
        

xg_reg = xgb.XGBRegressor(early_stopping_rounds = 10)

# Applying Grid Search to find the best model and the best parameters
#cross_v = GridSearchCV(xg_reg, parameters,scoring = 'neg_mean_squared_error',cv = 2)
#cross_v.fit(X_train,y_train)
#best_parameters = cross_v.best_params_

'''from functools import partial
from hypopt import GridSearch
grid_search = GridSearch(model = xg_reg, param_grid = parameters, cv_folds = 10)
grid_search = grid_search.fit(X_train, y_train, seller_val_set[0], seller_val_set[1],scoring = 'neg_mean_squared_error')
best_parameters = grid_search.get_params()
'''
    
xg_reg = xgb.XGBRegressor(early_stopping_rounds = 10)
best_parameters = {}
best_parameters['sublsample'] = 1
best_parameters['min_child_weight'] = 1
best_parameters['n_estimators'] = 10000
best_parameters['max_depth'] = 3
best_parameters['gamma'] =0
best_parameters['colsample_bytree'] = 1
best_parameters['lambda'] = 10
best_parameters['learning_rate'] =0.1
best_parameters['objective'] = 'count:poisson' 
best_parameters['eval_metric'] = 'rmse'
xg_reg = xgb.XGBRegressor(**best_parameters)
print(xg_reg.get_params)
eval_set = [(X_train,y_train),(X_val,y_val)]
xg_reg.fit(X_train,y_train,eval_set = eval_set,sample_weight = y_train_weights,early_stopping_rounds = 100)

preds = xg_reg.predict(X_val)
preds_list = list(preds.reshape(len(sellers),len(preds)//len(sellers)))
rmse = np.sqrt(mean_squared_error(np.nan_to_num(y_val), preds))/np.mean(y_val)
print("RMSE: %f" % (rmse))


X_train.shape
rmse = {}
for seller in sellers:
    rmse[seller] = np.sqrt(mean_squared_error((y_v_list[sellers.index(seller)]), preds_list[sellers.index(seller)]))/y_v_list[sellers.index(seller)].mean()
naive_score = {}
rmspe = {}
for seller in sellers:
    rmspe[seller] = np.sqrt(mean_squared_error((y_v_list[sellers.index(seller)]), preds_list[sellers.index(seller)]))/(y_v_list[sellers.index(seller)]).mean()

for seller in sellers:
    naive_score[seller] = moving_average_error[seller]/rmse[seller]
print(naive_score)
params = (pred_period,look_back_period,time_blocks)
naive_df = pd.DataFrame([naive_score,rmse,rmspe,moving_average_error],index = ['naive_score','rmse_model','rmspe_model','rmse_naive']).transpose()

preds = xg_reg.predict(X_val)
preds_list = list(preds.reshape(len(sellers),len(preds)//len(sellers)))
preds_train  = xg_reg.predict(X_train)
preds_train_list = list(preds_train.reshape(len(sellers),len(preds_train)//len(sellers)))


########################



##################
plt.clf()
residuals = y_val.ravel()-preds
plt.scatter(y_val,residuals)
##plots
import pylab 
pylab.clf()
pylab.bar(range(len(xg_reg.feature_importances_)), xg_reg.feature_importances_)
pylab.xticks(range(len(xg_reg.feature_importances_)),var_names)
pylab.show()

seller_pos = 0
seller_pos+=1
seller = sellers[seller_pos]
plt.clf()

plt.plot(dates[look_back_period:-2*pred_period],preds_train_list[seller_pos],color = 'greenyellow',label = 'seller {} train prediction'.format(str(seller)))
plt.plot(dates[look_back_period:-2*pred_period],y_t_list[seller_pos],color = 'g',label = 'seller {} train set'.format(str(seller)))
plt.plot(dates[-2*pred_period:-pred_period],preds_list[seller_pos],color = 'aqua',label = 'seller {} validation prediction'.format(str(seller)))
plt.plot(dates[-2*pred_period:-pred_period],y_v_list[seller_pos], color= 'b',label = 'seller {} validation set'.format(str(seller)))
plt.legend()
column ='daily_views_share'
complete_view.loc[(slice(None),seller),column].reset_index(level = 1)[column].tshift(periods = 0,freq = 'D').fillna(0).plot()

views.loc[(slice(None),seller),'daily_sales_sum'].reset_index(level = 1)['daily_sales_sum'].tshift(periods = 0,freq = 'D').fillna(0).plot()

view_seller.fillna('mean').tshift(periods = 2*pred_period, freq = 'D').rolling('{}d'.format(pred_period)).mean().daily_sales_sum.plot()

['date','amount_of_ads','active_seller','ad_type_mean','daily_views_sum','price_median','position_median','sold_quantity_sum','gini_ads','conversion','share','daily_sales_sum']
plt.clf()
for feature in ['amount_of_ads','price_median','position_median','daily_sales_sum']:
    feature = features.index(feature)
    plt.plot(X_dates,X[:,seller_pos,feature]/(max(X[:,seller_pos,feature])-min(X[:,seller_pos,feature]))-min(X[:,seller_pos,feature]),label = features[feature])
    plt.legend()
