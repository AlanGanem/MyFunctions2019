# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:17:14 2019

@author: PC10
"""

period = 30

date_format = "%Y-%m-%d"




final_date = datetime.datetime.now().strftime(date_format)
init_date = (datetime.datetime.now() - datetime.timedelta(days=period)).strftime(date_format)

data_marco = get_data(day='today',period = period)
clustered_data_marco = apply_product_id(data_marco)
clustered_data_marco = price_clustering(clustered_data_marco)[0]
ranking_marco = tratamento(clustered_data_marco,column = 'product_id_by_price')

ranking_marco.loc[:,ranking_marco.columns.isin(['product_name','product_id','father_category','amount_of_sellers','category_sold','top_sellers','top_ads','price_range','activity_ratio_top_1','product_min_price','product_max_price'])].to_csv('C:\ProductClustering\output_data\csv_files\product_rankings\\products_{}_{}1.csv'.format(init_date,final_date))
clustered_data_marco.loc[:,clustered_data_marco.columns.isin(['min_date','max_date','ad_title','ad_id','seller_id','sold_difference','price_min','price_max','activity_ratio','product_id','product_id_by_price'])].to_csv('C:\ProductClustering\output_data\csv_files\clustered_ads\\ads_{}_to_{}.csv'.format(init_date,final_date))