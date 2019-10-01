# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 01:03:55 2019

@author: PC10
"""


import psycopg2
import numpy as np
import pandas as pd

def updatetable(data,  column = 'product_id' ):
    try:
        data_ids = data[[column,'index']].values
        connection = psycopg2.connect("host='192.168.25.68' dbname='postgres' user='postgres' password='jg.210795'")
        cursor = connection.cursor()

        sql_update_query = """Update ads_scd set {} = {} where id = {}"""
        len_ = len(data_ids)
        i=0
        for product_id, index in data_ids:
            connection.rollback()
            cursor.execute(sql_update_query.format(column,product_id, index))
            connection.commit()
            i+=1
            print('[{}/{}]'.format(i,len_))
        
    except (Exception, psycopg2.Error) as error:
        print("Error in update operation", error)
    finally:
        # closing database connection.
        
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
