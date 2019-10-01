# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:15:19 2019

@author: PC10
"""
import numpy as np
from func_gini import gini
from func_density_plot import density_plot
from func_price_clustering import price_clustering
import wquantiles as wq
from scipy import stats
import sklearn
from func_get_product_history import get_product_history
import pandas as pd
import tqdm
import datetime
import TimeSeriesUtils as TSU

    
    
def get_and_prepare_product_data(product_id, min_price,max_price, title_ilike,title_not_ilike,  features,  dependent_variable,drop_blackout = False):
    assert features[-1:] == dependent_variable

    history = get_product_history(product_id = product_id, min_price =min_price, max_price = max_price ,drop_blackout = drop_blackout, title_ilike=title_ilike, title_not_ilike =title_not_ilike )
    history['date'] = pd.to_datetime(history['date'], errors  = 'coerce',format = '%Y-%m-%d')
    #history = history[(history.daily_sales > min(history.daily_sales))&(history.daily_sales < max(history.daily_sales))]
    
    sellers = list(history.groupby('seller_id').sum()[history.groupby('seller_id').sum().daily_sales > min_sold].daily_sales.index)
    history = history[history['seller_id'].isin(sellers)]
    
    len_= len(history)
    daily_views = []
    for (date, ad) in tqdm.tqdm(history[['date','ad_id']].values):
        try: 
            daily_views.append(max(history[(history.date== date)&(history.ad_id ==ad)].views.max()-history[(history.date== (date  - datetime.timedelta(1)))&(history.ad_id ==ad)].views.max(),0)) 
        except KeyError:
            daily_views.append(np.nan)
            
    history = history.assign(daily_views = daily_views)
    history = history.fillna(method = 'backfill')
    history = history.fillna(0)[(stats.zscore(history.fillna(0)['daily_sales']) < 10)]
    history = history[history.daily_sales >=0]
    history.daily_sales.max()
    
    view = price_clustering(history[history.daily_sales > 0 ],column_name = 'price', fluctuation = 0.2)
    
    
    history = history.dropna()
    history_filtered = history[history.seller_id.isin(sellers)]
    sellers_dates = {seller:{'initial_date':TSU.timestamp_to_datetime(history_filtered[history_filtered['seller_id'] == seller]['date'].nsmallest(2).max()),'final_date':TSU.timestamp_to_datetime(history_filtered[history_filtered['seller_id'] == seller]['date'].nlargest(2).min())} for seller in  sellers}
    history_filtered =history_filtered.assign(active_seller = 0)
    TSU.timestamp_to_datetime(history_filtered['date'].max())
    for seller in sellers:
        initial_date = sellers_dates[seller]['initial_date']
        final_date= sellers_dates[seller]['final_date']    
        history_filtered[(history_filtered['seller_id'] == seller)&(history_filtered['date'].isin(pd.date_range(initial_date,final_date)))] = history_filtered[(history_filtered['seller_id'] == seller)&(history_filtered['date'].isin(pd.date_range(initial_date,final_date)))].assign(active_seller = 1, inplace = True)
    
    def fu(x):
        d={}
        d['date'] = x['date'].max()
        d['active_seller']  = x['active_seller'].max()
        d['amount_of_ads'] = x['active_seller'].count()
        d['category_id'] = sklearn.utils.extmath.weighted_mode(x['category_id'],np.nan_to_num(x['daily_sales']))[0].max()
        d['daily_sales_sum'] = x['daily_sales'].sum()
        try:
            d['ad_type_mean'] = np.average(x['ad_type_id'],weights = np.nan_to_num(x['daily_sales']))
        except:
            d['ad_type_mean'] = x['ad_type_id'].mean()
        d['daily_views_sum'] = x['daily_views'].sum()
        
        if x['daily_sales'].max()> 0:
            d['price_median'] = wq.quantile(x['price'],x['daily_sales'],0.5)
        else:
            d['price_median'] = wq.quantile(x['price'],len(x['price'])*[1],0.5)
        if np.isnan(d['price_median']):
            d['price_median'] = np.median(d['price_median'])
        if x['daily_sales'].max() > 0:
            d['position_median'] = wq.quantile(x['position'],len(x['price'])*[1],0.5)
        else:
            d['position_median'] = wq.quantile(x['position'],x['daily_sales'],0.5)
        if np.isnan(d['position_median']):
            d['position_median'] = np.median(d['position_median'])
        
        d['sold_quantity_sum'] = x['sold_quantity'].sum()
        d['gini_ads'] = gini(x['daily_revenues'].values)
        if x['daily_views'].sum() > 0:
            d['conversion'] = x['daily_sales'].sum()/x['daily_views'].sum()
        else:
            d['conversion'] = 0
        d['share'] = x['daily_revenues'].sum()/x['market_size'].max()
        if np.isnan(d['share']):
            d['share'] = 0
        return (pd.Series(d))
    
    market_sizes = [history_filtered.groupby('date').get_group(i).daily_revenues.sum() for i in history_filtered.date.unique()]
    
    dflist=[]
    i = 0
    for date in history['date'].unique():
        dflist.append(history_filtered[history_filtered.date == date].assign(market_size = market_sizes[i].max()))
        i+=1
    
    history_filtered = pd.concat(dflist)
    groupped = history_filtered.groupby('date')
    
    
    cnn_X_shape = (len(sellers),len(features)-1)
    cnn_y_shape = (len(sellers),1)
    gabarito = pd.DataFrame(np.zeros((len(sellers),len(features))), columns = features,index = sellers)
    
    date_interval = pd.date_range(min(sorted(history_filtered.date.unique())),max(sorted(history_filtered.date.unique())))
    
    dates = {}
    for date in tqdm.tqdm(date_interval):
        date = TSU.timestamp_to_datetime(date)
        try:
            data = groupped.get_group(date).reset_index(drop=False).groupby('seller_id').apply(fu)[features]
            df = gabarito.assign(date = date).copy()
            df.loc[data.index] = data.assign(date = date)
        except KeyError as error:        
            df = gabarito.assign(date=date).copy()
        dates[TSU.timestamp_to_datetime(date)] = df    
    
    min_date = min(sorted(list(dates.keys())))
    max_date = max(sorted(list(dates.keys())))
    
    lista_X = []
    for key in sorted(list(dates.keys())):
        try:
            lista_X.append(dates[key][features].values)
        except:
            print(date)
            lista_X.append(np.zeros(cnn_X_shape))    
    X = np.array(lista_X)
    return X, sellers, features 

