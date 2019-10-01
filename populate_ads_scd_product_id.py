# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:03:58 2019

@author: PC10
"""
from func_update_ads_scd import update_ads_scd
from func_updatetable import updatetable

i = 0
while i <= 10:
    update_ads_scd(path = r'C:\ProductClustering\productsDB\products_db_objects\seeds\\',dic_name = 'seeds_2019_02_04', update_db=False)
    i+=1