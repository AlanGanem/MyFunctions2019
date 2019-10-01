# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:35:24 2019

@author: PC10
"""

import psycopg2
import pandas.io.sql as sqlio
import datetime

def get_data(free_query= None, period=7,final_date_string ='today',date_format = "%Y-%m-%d", query_path='C:\ProductClustering\sql_queries\categories_metrics.txt'):
    conn = psycopg2.connect("host='192.168.25.68' dbname='postgres' user='postgres' password='jg.210795'")
    if free_query is None:
        days_size = period    
        if not final_date_string == 'today':
            dt = datetime.datetime.strptime(final_date_string,date_format) 
            final_date = datetime.datetime(dt.year,dt.month,dt.day).strftime(date_format)
            init_date = (dt - datetime.timedelta(days=days_size)).strftime(date_format)
        else:
            final_date = datetime.datetime.now().strftime(date_format)
            init_date = (datetime.datetime.now() - datetime.timedelta(days=days_size)).strftime(date_format)
        print(init_date + ' to '+final_date)
        f = open(query_path)
        query = f.read()
        query = query.format(init_date,final_date)
    else:
        query = free_query
        
    dat = sqlio.read_sql_query(query, conn)
    conn = None
    return dat