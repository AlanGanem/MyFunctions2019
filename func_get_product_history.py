# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 01:07:44 2019

@author: PC10
"""

import datetime
from func_get_data import get_data

def get_product_history(product_id, initial_date = None,min_price=None,max_price=None, title_ilike = None,title_not_ilike = None, grouped = False,drop_blackout = True):
    
    if initial_date is None:
        initial_date = "2000-01-01"
    if min_price is None:
        min_price = 0
    if max_price is None:
        max_price =9999999
    
    if isinstance(title_ilike,list):
        title_ilike_i = ''
        i = 0
        for text in title_ilike:
            if text != '':
                if i ==0:
                    title_ilike_i += ''' and (ads_scd.ad_title ilike '%{}%' '''.format(text)
                    i+=1
                else:
                    title_ilike_i += ''' or ads_scd.ad_title ilike '%{}%' '''.format(text)
        if title_ilike_i != '':
            title_ilike =  title_ilike_i+')'
        else:
            title_ilike = ''
            
    elif isinstance(title_ilike,str):
        if title_ilike != '':
            title_ilike = '''and ads_scd.ad_title ilike '%{}%' '''.format(title_ilike)
        else:
            title_ilike = ''
    else:
        title_ilike = ''
    
    if isinstance(title_not_ilike,list):
        title_not_ilike_i = ''
        i = 0
        for text in title_not_ilike:
            if text != '':
                if i ==0:
                    title_not_ilike_i += ''' and (ads_scd.ad_title not ilike '%{}%' '''.format(text)
                    i+=1
                else:
                    title_not_ilike_i += ''' or ads_scd.ad_title not ilike '%{}%' '''.format(text)
        if title_not_ilike_i != '':
            title_not_ilike =  title_not_ilike_i+')'
        else:
            title_not_ilike =''
                          
    elif type(title_not_ilike) is str:
        if title_not_ilike != '':
            title_not_ilike = '''and ads_scd.ad_title not ilike '%{}%' '''.format(title_not_ilike)
        else:
            title_not_ilike = ''
    else:
        title_not_ilike = ''

    if isinstance(product_id,list):
        product_id = str(product_id)
        product_id = ' in ' + '('+product_id[1:-1]+')' 
    elif type(product_id) is int:
        product_id = ' = '+ str(product_id)
    else:
        raise TypeError('product_id must an integer or a list of integers')
    
    if drop_blackout:
        drop_blackout  = "and date not between '2018-07-20' and '2018-11-13'"
    else:
        drop_blackout = ''
    
    if grouped:
        data = get_data(free_query = '''    
        select date, string_agg(distinct cast(product_id as text), ',') ,
        json_object_agg(history.ad_id,(ads_scd.ad_title,daily_sales,price,ads_scd.ad_type_id, seller_id,ads_scd.product_id,date)) as ad_dict,
        sum(views)/greatest(1,count(distinct seller_id)) as views_per_seller,
        count(distinct seller_id) as amount_of_sellers,
        sum(daily_sales) as daily_sales,
        sum(daily_sales*price) as daily_revenues,
        sum(sold_quantity) as sold_quantity,
        1.0*sum(daily_sales)/greatest(count(distinct seller_id),1) as daily_sales_per_seller,
        cast(avg(position) as integer) as avg_position,
        min(position) as best_position ,
        cast(sum(price*greatest(0,daily_sales))/greatest(1,sum(daily_sales)) as integer) sales_weighted_avg_price,
        percentile_disc(0.05) WITHIN GROUP (ORDER BY history.price) as price_min,
        percentile_disc(0.5) WITHIN GROUP (ORDER BY history.price) as price_median,
        percentile_disc(0.95) WITHIN GROUP (ORDER BY history.price) as price_max,
        count(distinct history.ad_id) amount_of_ads
        from history
        inner join ads_scd on history.ad_id = ads_scd.ad_id
        inner join ad_static on ads_scd.ad_id = ad_static.ad_id
        where ads_scd.product_id {} and date >'{}' and date not between '2018-07-20' and '2018-11-13' and history.price > {} and history.price < {} {} {}
        group by date      
        '''.format(product_id, initial_date,min_price,max_price,title_ilike, title_not_ilike))
    
    else:
        
        data = get_data(free_query = '''select 
            date as date,
            history.ad_id as ad_id,
            ads_scd.ad_title as ad_title,
            history.daily_sales as  daily_sales,
            ads_scd.ad_type_id as ad_type_id, 
            ad_static.seller_id,
            ads_scd.product_id, 
            history.views as views,
            history.daily_sales*history.price as daily_revenues,
            history.sold_quantity ,
            history.position ,
            history.price as price,
            ad_static.category_id as category_id,
            seller_scd.seller_power_id  as seller_power,
            seller_scd.seller_state  as seller_state
    		
            from history
            
            left join ads_scd on history.ad_id = ads_scd.ad_id and history.date >= ads_scd.initial_date and history.date < ads_scd.final_date  
            left join ad_static on ads_scd.ad_id = ad_static.ad_id
            left join seller_scd on ad_static.seller_id = seller_scd.seller_id and history.date >= seller_scd.initial_date and history.date < seller_scd.final_date  

            where ads_scd.product_id {} and date >'{}' {} and history.price > {} and history.price < {} {} {} '''.format(product_id, initial_date, drop_blackout,min_price,max_price,title_ilike, title_not_ilike))
        

    if all([i in data.columns for i in ['daily_sales','sold_quantity','views']]):
        view = data.set_index(['date','ad_id']).groupby(level = 1)[['sold_quantity']].diff().reset_index(drop  = True)
        data = data.assign(daily_sales = view.values.flatten())
        view = data.set_index(['date','ad_id']).groupby(level = 1)[['views']].diff().fillna(0).reset_index(drop  = True)        
        view[view < 0] = 0
        data = data.assign(daily_views = view.values.flatten())
        data = data.assign(daily_revenues = data['daily_sales']*data['price'])
    return data
