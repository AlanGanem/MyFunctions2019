import pandas as pd
import pickle
import os
import psycopg2
import pandas.io.sql as sqlio
import tqdm

def get_father_categories():
    
    conn = psycopg2.connect("host='192.168.25.68' dbname='postgres' user='postgres' password='jg.210795'")
    query = 'select * from category3'
    cat_tree = sqlio.read_sql_query(query, conn)
    query2 = 'select distinct(category_0) from category3'
    father_cats = sqlio.read_sql_query(query2, conn)
    father_cats = father_cats.category_0.str.replace('MLB','')
    query3 = 'SELECT * FROM public.category where category.category_id in {}'.format(tuple(father_cats.values))
    father_cats = sqlio.read_sql_query(query3, conn)
    
    conn = None   
    for column in list(cat_tree):
        cat_tree = eval('cat_tree.assign(%s = cat_tree.%s.str.replace("MLB",""))' %(column,column))
    i = 0
    all_lvls = {}
    for column in tqdm.tqdm(list(cat_tree)[::-1]):
        nonans = cat_tree[column][cat_tree[column]>'0']
        indexes = []
        for ad in tqdm.tqdm(nonans):
            a = cat_tree[cat_tree[column] == ad]
            all_lvls[ad] = a.category_0.max()
            indexes.append(a.index.max())
            i+=1
        cat_tree = cat_tree.drop(indexes)
     
    f = open(os.path.join(os.path.dirname(r'C:\ProductClustering\input_Data\\'), 'cat_dic'),'wb')
    pickle.dump(all_lvls,f)
        
    return all_lvls

