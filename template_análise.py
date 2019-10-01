############################ importing #####################################
import sys
sys.path.append(r'C:\ProductClustering\my_functions')

import pandas as pd
import time

from func_price_clustering import price_clustering
from func_tratamento import tratamento
from func_apply_product_id  import apply_product_id
from func_get_data import get_data
############################## processing ################################

data = get_data(period = 30)
data = data.dropna()
period_name = str(data.date_min.min())+'_to_'+str(data.date_max.max())
clustered_data = apply_product_id(data, update_db = False)
clustered_data = clustered_data.dropna()
clustered_data = price_clustering(clustered_data, column ='product_id')[0]
ranking = tratamento(clustered_data,column = 'product_id')

ranking[['product_name','product_id_by_price','product_id','father_category','amount_of_sellers','product_sold','top_sellers','top_ads','product_min_price','product_max_price','activity_ratio_top_1']].to_csv('C:\ProductClustering\output_data\csv_files\product_rankings\\products_{}.csv'.format(period_name))
clustered_data[['date_min','date_max','ad_title','ad_id','seller_id','sold_difference','price_min','price_max','activity_ratio','product_id','product_id_by_price']].to_csv('C:\ProductClustering\output_data\csv_files\clustered_ads\\ads_{}.csv'.format(period_name))


########################### plot ##################################3
from func_get_product_history import get_product_history
import pandas as pd
import numpy as np
a = class_products_db_finder.products_db_finder()
a.init_products_db(prod_db)

product_name = 'banheiro quimico portatil camping'
similar = a.get_similar_products(title= product_name)
history = get_product_history(product_id = [9550,9549], min_price = 150, max_price=800)

history.to_csv(r'C:\Users\PC10\Desktop\Alan\share_analysis\{}.csv'.format(product_name))

view = history[history.seller_id == 97556253][history.date.astype(str).str.contains('2018-12')].daily_sales.sum()

np.__name__

'''view = pd.DataFrame(columns=['ad_title','daily_sales','price','ad_type','seller_id','product_id','date'])
for date in history.date.astype(str):
    view = view.append(pd.DataFrame(history[history.date.astype(str) == date].max().ad_dict).transpose().rename(columns ={'f1':'ad_title','f2':'daily_sales','f3':'price','f4':'ad_type','f5':'seller_id','f6':'product_id','f7':'date'})) 
our_ads = view[view.seller_id == 97556253]
our_ads = our_ads.assign(daily_revenues = our_ads.daily_sales*our_ads.price)
revenues = our_ads.groupby('date').apply(lambda x:  (x['daily_sales']*x['price']).sum()).reset_index(drop = False).rename(columns= {0:'daily_revenues_our_ads'})
if len(revenues) == 0:
    revenues  = pd.DataFrame(columns = ['date','daily_revenues_our_ads'] )
dates = history[~history.date.astype(str).isin(np.intersect1d(revenues.date.astype(str), history.date.astype(str)))].date.astype(str).values
for i in range(len(dates)):
    revenues=revenues.append(pd.DataFrame({'date':[dates[i]],'daily_revenues_our_ads':[0]}), ignore_index=True)

assert len(revenues)==len(history)

revenues.daily_revenues_our_ads.sum()/history.daily_revenues.sum()'''

#revenues.to_csv(r'C:\Users\PC10\Desktop\Alan\share_analysis\historico_{}_our_ads.csv'.format(product_name))

view[view.date == '2019-03-11'].seller_id
######################### sales leaders ######################
