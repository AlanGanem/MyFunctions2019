# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 01:30:55 2019

@author: PC10
"""
import pickle
import os
import pandas as pd
from class_products_db_finder import products_db_finder

def apply_product_id(data,min_sim = 0.90,threshold = 0.3, path = r'C:\ProductClustering\productsDB\products_db_objects\\',obj_name = 'products_db_object', dic_name = 'products_db_dict', update_db = False, update_linkage =False,clustering_algorithm = 'agglomerative'):
    #import  product db
    print('importing DB dictionary')
    g = os.path.join(os.path.dirname(path), dic_name)
    prod_db_dic = pickle.load(open(g, 'rb'))
    prod_db = pd.DataFrame(prod_db_dic)
    a = products_db_finder()
    a.init_products_db(prod_db)
    #apply product  labels to data
    print('looking for matching products in  DB')
    clustered_data = a.search_engine(threshold = threshold, min_sim = min_sim,raw_df = data, centroids = prod_db,clustering_algorithm = clustering_algorithm,min_amount_analogous= 3)
    #udpate products_db
    if update_db:  
        print('updating DB')
        a.update_products_db(new_existing_products = a.new_existing_products, new_products=a.new_products, update_linkage = update_linkage)
        print('exporting db dictionary')
        a.export_db_dic(path = path, file_name = dic_name)
    
    return data.assign(product_id = clustered_data['product_id'])

