# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:42:12 2019

@author: PC10
"""
import pandas as pd
from scipy import stats
import TimeSeriesUtils as TSU
from TimeSeriesUtils import fu
from func_slope import func_slope
from func_get_product_history import get_product_history
import numpy as np
from scipy.stats import variation
from functools import partial

min_sold = 30
product_id = [9550]
min_price = 250
max_price = 600
features = ['market_size','price_min','position_max','amount_of_ads','daily_views_sum','price_median','position_median','sold_quantity_sum','gini_ads','conversion','daily_sales_sum']
one_hot_features = ['seller_state','seller_power']
pred_period = 40
look_back_period = 20
time_blocks = 1
foward_pred_goal = pred_period
drop_blackout = False
seller_val= None

history = get_product_history(product_id = product_id, min_price =min_price, max_price = max_price,drop_blackout= drop_blackout,title_ilike = [''],title_not_ilike =[''])

sellers = list(history.groupby('seller_id').sum()[history.groupby('seller_id').sum().daily_sales > min_sold].daily_sales.index)
history = history[history['seller_id'].isin(sellers)]
history = history.fillna(method = 'backfill')
history = history[history.daily_sales >=0]
history = history.fillna(0)[(stats.zscore(history.fillna(0)['daily_sales']) < 3)]
history.daily_sales.max()

history_filtered = history[history.seller_id.isin(sellers)]

sellers_dates = {seller:{'initial_date':history_filtered[history_filtered['seller_id'] == seller]['date'].min(),'final_date':history_filtered[history_filtered['seller_id'] == seller]['date'].max()} for seller in  sellers}
history_filtered =history_filtered.assign(active_seller = 0)    

for seller in sellers:
    initial_date = sellers_dates[seller]['initial_date']
    final_date= sellers_dates[seller]['final_date']    
    history_filtered[(history_filtered['seller_id'] == seller)&(history_filtered['date'].isin(pd.date_range(initial_date,final_date)))] = history_filtered[(history_filtered['seller_id'] == seller)&(history_filtered['date'].isin(pd.date_range(initial_date,final_date)))].assign(active_seller = 1, inplace = True)


assert isinstance(one_hot_features,list)
history_filtered = pd.get_dummies(history_filtered,columns = one_hot_features)
one_hot_feature_list = []
for feature in one_hot_features:
     one_hot_feature_list+=[column for column in history_filtered.columns  if feature  in column]

metrics = history_filtered.set_index(['date','seller_id']).groupby(level = 'date').apply(lambda x: x.assign(market_daily_views  = x['daily_views'].sum(),market_median_price = x['price'].quantile(0.25),market_size = (x['daily_sales']*x['price']).sum(),market_size_units = x['daily_sales'].sum(),active_seller = 1).groupby(level = 1).apply(partial(TSU.fu,one_hot_feature_list = one_hot_feature_list)))
metrics =metrics.groupby(level = 'date').apply(lambda x: x.assign(amount_of_ads_total = x['amount_of_ads'].sum(),amount_of_sellers =  x['active_seller'].sum()))
rolling_revenues =metrics.groupby(level = 'seller_id')[['daily_revenues_sum','market_size']].apply(lambda x: x.rolling(look_back_period).mean())
rolling_revenues = rolling_revenues.groupby(level = 'date').apply(lambda x: x.assign(market_size = x['market_size'].max()))
rolling_diffs =rolling_revenues.rename(columns =  {'daily_revenues_sum':'rolling_daily_revenues_diff','market_size':'rolling_market_size_diff'}).groupby(level = 'seller_id')[['rolling_daily_revenues_diff','rolling_market_size_diff']].apply(lambda x: x.diff())
rolling_revenues[rolling_revenues<1e-6]=0
rolling_revenues = rolling_revenues.rename(columns =  {'daily_revenues_sum':'rolling_daily_revenues_mean','market_size':'rolling_market_size'}).assign(rolling_share = rolling_revenues['daily_revenues_sum']/rolling_revenues['market_size']).sort_index()
metrics = pd.concat([metrics,rolling_revenues], axis = 'columns')
metrics = pd.concat([metrics,rolling_diffs], axis = 'columns')
dates = list(pd.DatetimeIndex(pd.date_range(start = metrics.reset_index().date.min(),end = metrics.reset_index().date.max())))
idx = pd.MultiIndex.from_product([dates,sellers],names = ['date','seller_id'])
complete_view = pd.concat([pd.DataFrame(index= idx),metrics],axis = 'columns')
complete_view = complete_view.groupby(level = 1).apply(lambda x: x.sort_index().assign(day = np.arange(len(pd.date_range(start = metrics.reset_index().date.min(),end = metrics.reset_index().date.max())))))
complete_view = complete_view.groupby(level = 'seller_id').apply(lambda x: x.assign(month = x.index.levels[0].month,year = x.index.levels[0].year))
complete_view = complete_view.groupby(level = 'seller_id').apply(lambda x: x.assign(daily_sales_cumsum = x[['daily_sales_sum']].fillna(0).cumsum()))
complete_view['rolling_share'] = complete_view['rolling_share']*1
complete_view['daily_sales_sum'] = complete_view['daily_sales_sum']*1
complete_view['rolling_market_size'].max()
for seller in sellers:
    initial_date = sellers_dates[seller]['initial_date']
    final_date= sellers_dates[seller]['final_date']    
    date_range = pd.date_range(initial_date,final_date)
    for date in date_range:
        complete_view.loc[(date,seller),'active_seller'] = 1


plt.clf()
for i in range(len(sellers)):
    complete_view.loc[(slice(None),sellers[i]),'daily_sales_cumsum'].plot()












column = 'daily_views_share'
complete_view.loc[(slice(None),slice(None)),column]

seller_invariant_dynamic_vars = ['market_median_price','market_size','amount_of_sellers']
static_vars = ['gini_ads','relative_price','amount_of_ads','active_seller']
dynamic_vars = ['daily_views_sum','rolling_share','daily_sales_sum']


dates = sorted(complete_view.reset_index()['date'].unique()) 
seller_variant = [complete_view[dynamic_vars].loc[(date,slice(None)),:].fillna(0).values for date in dates]
funcs = ['mean','slope']

X_seller_variant = np.array(seller_variant)
X_seller_variant.shape

seller_invariant = [complete_view[seller_invariant_dynamic_vars].loc[(date,slice(None)),:].fillna(0).max()[seller_invariant_dynamic_vars].values for date in dates]


X_seller_invariant = np.array(seller_invariant)
X_seller_invariant = X_seller_invariant.reshape(X_seller_invariant.shape[0],1,X_seller_invariant.shape[1])
X_seller_invariant.shape

static = [complete_view[static_vars].loc[(date,slice(None)),:].fillna(0)[static_vars].values for date in dates]
X_static = np.array(static)
X_static.shape

seller_variant_names = []
for seller in sellers:
    for var in dynamic_vars:    
        for func in funcs:
            seller_variant_names.append('{}_{}_{}'.format(seller,var,func))
seller_invariant_names = []
for var in seller_invariant_dynamic_vars:
    for func in  funcs:
        seller_invariant_names.append('{}_{}'.format(var,func))
seller_static_names = []
for seller in sellers:
    for var in static_vars:
        seller_static_names.append('{}_{}'.format(seller,var))
dummies_names = [str(seller) for seller in sellers]
var_names =  dummies_names+seller_static_names+['active']+seller_variant_names+ seller_invariant_names

X_t_list = []
X_v_list = []
y_t_list = []
y_v_list = []

pooled_output = True
for seller in sellers:
    seller_active = complete_view.loc[(slice(None),seller),'active_seller'].fillna(0).values.reshape(-1,1)
    
    seller_active_train = seller_active[look_back_period:-2*pred_period]
    seller_active_val = seller_active[-2*pred_period:-1*pred_period]
    y_seller_axis = sellers.index(seller)    
    
    X_train_seller_variant, y_train, X_val_seller_variant, y_val = TSU.chunk_to_pooled_2d(X_seller_variant,pred_period=pred_period,look_back_period=look_back_period,y_seller_axis = y_seller_axis , output_index = -1,output_axis =-1, time_blocks = time_blocks, functions = [np.mean,func_slope],pooled_output = pooled_output )
    X_train_seller_invariant, _ , X_val_seller_invariant, _ = TSU.chunk_to_pooled_2d(X_seller_invariant,pred_period=pred_period,look_back_period=look_back_period,y_seller_axis = y_seller_axis , output_index = -1,output_axis =-1, time_blocks = time_blocks, functions = [np.mean,func_slope],pooled_output = pooled_output )
    
    X_train = np.concatenate([seller_active_train,X_train_seller_variant,X_train_seller_invariant],axis = 1)    
    X_train_static = X_static[:X_train.shape[0]]    
    X_train_static = X_train_static.reshape(X_train_static.shape[0],X_train_static.shape[1]*X_train_static.shape[2])
    
    X_val = np.concatenate([seller_active_val,X_val_seller_variant,X_val_seller_invariant],axis = 1)
    X_val_static = X_static[X_train.shape[0]:X_train.shape[0]+X_val.shape[0]]
    X_val_static = X_val_static.reshape(X_val_static.shape[0],X_val_static.shape[1]*X_val_static.shape[2])
    
    t_dummies = np.zeros((X_train.shape[0],len(sellers)))
    t_dummies[:,y_seller_axis] =  1
    v_dummies = np.zeros((X_val.shape[0],len(sellers)))
    v_dummies[:,y_seller_axis] =  1
    
    print(v_dummies[0])    
    X_train, X_val = np.concatenate([t_dummies,X_train_static,X_train],axis = 1), np.concatenate([v_dummies,X_val_static,X_val],axis= 1)
    if pooled_output:
        y_train,y_val = np.average(y_train,axis = 1),np.average(y_val,axis = 1)
    else:
        pass

    X_t_list.append(X_train)
    y_t_list.append(y_train)
    X_v_list.append(X_val)
    y_v_list.append(y_val)

X_train = np.array(X_t_list)
X_train= X_train.reshape((X_train.shape[0]*X_train.shape[1],)+X_train.shape[2:])

y_train= np.array(y_t_list)
y_train = y_train.reshape((y_train.shape[0]*y_train.shape[1],)+y_train.shape[2:])

X_val_15 = np.array(X_v_list)[:,:foward_pred_goal,:]
X_val = np.array(X_v_list)
X_val_15 = X_val_15.reshape((X_val_15.shape[0]*X_val_15.shape[1],)+X_val_15.shape[2:])
X_val = X_val.reshape((X_val.shape[0]*X_val.shape[1],)+X_val.shape[2:])

y_val_15 = np.array(y_v_list)[:,:foward_pred_goal]
y_val = np.array(y_v_list)
y_val_15 = y_val_15.reshape((y_val_15.shape[0]*y_val_15.shape[1],)+y_val_15.shape[2:])
y_val = y_val.reshape((y_val.shape[0]*y_val.shape[1],)+y_val.shape[2:])


y_train_weights = 1/(1+np.log1p(y_train))

assert len(var_names) == X_train.shape[1]

