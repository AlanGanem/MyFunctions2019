# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:54:50 2019

@author: PC10
"""

import matplotlib.pyplot as plt
import cluster_ousado
from func_get_data import get_data
import pandas as pd
import pickle
from func_apply_word_embedings import apply_word_embedings
model_name = 'model_fast_text_sg_40'
seed_name_export = 'seeds_fast_text_2'


a = products_db_finder(model_name = model_name)

data = get_data(period = 360,final_date_string ='today',date_format = "%Y-%m-%d", query_path='C:\ProductClustering\sql_queries\categories_metrics.txt')
cluster_graph = a.graph_communities(data[['ad_id','ad_title','category_id','date_min','date_max']])
g1 = a.group_by_product(cluster_graph)
link = a.linkage(g1)
g1.word_vector

g1 = a.group_by_product(apply_word_embedings(pd.read_csv(r'C:\ProductClustering\productsDB\classtreste.csv')))
link  = pickle.load(open(r'C:\ProductClustering\productsDB\model_fast_text_sg_40_linkage','rb'))

g1 = a.group_by_product(cluster_graph)
hierarchy = g1.assign(product_id = a.hierarchycal_clustering(link,threshold = 0.3))
hierarchy[hierarchy.counter > 2].product_id.nunique()
g2 = a.group_by_product(hierarchy)
g3  = g2[g2.counter > 2]

a.export_db_dic('C:\ProductClustering\productsDB\products_db_objects\seeds\\',seed_name_export)

teste = a.get_similar_products(product_id =123,top_n =  5, title = ['camisa de futebol'])

range_= np.linspace(0,2,50)
sizes = {}
sizes['>=1'] = []
sizes['>2'] = []
sizes['>4'] = []
sizes['>8'] = []
for i in range_: 
    hierarchy = g1.assign(product_id = a.hierarchycal_clustering(link,threshold = i))
    sizes['>=1'] += [hierarchy[hierarchy.counter >= 1].product_id.nunique()]
    sizes['>2'] += [hierarchy[hierarchy.counter > 2].product_id.nunique()]
    sizes['>4'] += [hierarchy[hierarchy.counter > 4].product_id.nunique()]
    sizes['>8'] += [hierarchy[hierarchy.counter > 8].product_id.nunique()]
plt.clf()
for key,value in sizes.items():
    plt.plot(range_, value)
    
hierarchy = g1.assign(product_id = a.hierarchycal_clustering(link,threshold = 1))

g2 = a.group_by_product(hierarchy)
g3  = g2[g2.counter > 2]
a.init_products_db(g3)
a.export_db_dic('C:\ProductClustering\productsDB\products_db_objects\seeds\\',seed_name_export)
