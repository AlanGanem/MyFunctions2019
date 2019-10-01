
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:15:49 2019

@author: PC10
"""
import sys
sys.path.append(r'C:\ProductClustering\my_functions')
from func_search_engine import search_engine_fasttext
from func_get_data import get_data
import pandas as pd
import pickle
from func_apply_word_embedings import apply_word_embedings
from func_updatetable import updatetable
import class_products_db_finder
from func_search_engine import search_engine_fasttext
import os

def update_ads_scd(path = r'C:\ProductClustering\productsDB\products_db_objects\\',column = 'product_id',dic_name = 'products_db_dict',db_save_file_name = 'products_db',model_name = 'model_fast_text_sg_40',update_db=False, min_sim =0.90, condition = 'is null'):
    
    
    print('querying data')
    data = get_data(free_query = ''' select ad_static.category_id as category_id,initial_date as date_min,final_date as date_max,ads_scd.ad_id, ads_scd.id as index,ads_scd.ad_title
    from ads_scd 
    inner join ad_static on ads_scd.ad_id = ad_static.ad_id 
    where {} {} 
    limit(1000000)'''.format(column,condition))
    
    print ('{} ads found'.format(str(len(data))))
    
    data = data.dropna(subset=['ad_title'])
    assert len(data)>0
    data = apply_word_embedings(data, model_name = model_name)
    #data = data[data['ad_title_corpus'] != '']
    if len(data)<=0:
        print('all queryied titles are empty')
        return
    g = os.path.join(os.path.dirname(path), dic_name)
    prod_db = pd.DataFrame(pickle.load(open(g, 'rb')))
    data = data.astype({'date_min':str,'date_max':str})
    print('running search engine')
    if update_db:
        
        a = class_products_db_finder.products_db_finder(model_name = model_name )
        a.init_products_db(prod_db)
        applied_product_ids = a.search_engine(data,prod_db,min_sim = min_sim ,pre_computed_word_vectors = True)
        
        
        a.update_products_db(new_products = a.new_products, new_existing_products = a.new_existing_products)
        a.export_db_dic(path = r'C:\ProductClustering\productsDB\products_db_objects\\', file_name = db_save_file_name)
        applied_product_ids =applied_product_ids.rename(columns = {'product_id':'{}'.format(column)})
    else:
        applied_product_ids = search_engine_fasttext(data,prod_db,model_name=model_name, min_sim = min_sim)
        applied_product_ids =applied_product_ids.rename(columns = {'product_id_fasttext':'{}'.format(column)})
        print('{} ads uncategorized ({} = -1)'.format(str(len(applied_product_ids[applied_product_ids[column] ==-1])),column))
        print('{} different existing products found in {} categories'.format(str(applied_product_ids[applied_product_ids[column] != -1][column].count()),str(applied_product_ids[applied_product_ids[column] != -1][column].nunique())))
    
    eval('updatetable(data.assign({} = applied_product_ids[column]),column = column)'.format(column))
    
    return

