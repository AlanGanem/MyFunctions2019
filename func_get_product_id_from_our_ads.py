# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:03:07 2019

@author: PC10
"""
from func_get_data import get_data
def get_product_id_from_our_ads(title):
    return get_data(free_query = '''
    select ad_static.ad_title,ads_scd.product_id,ads_scd.product_id_2 from ad_static
    join ads_scd on ad_static.ad_id = ads_scd.ad_id
    where ads_scd.final_date > '2020-01-01' and ad_static.seller_id = 97556253 and ad_static.ad_title ilike '%inflador%'
    ''')
    
    