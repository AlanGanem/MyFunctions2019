# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:34:59 2019

@author: PC10
"""

from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
init_date = history.date.min().date()
final_date = history.date.max().date()
search = 'pipoqueira eletrica'

googletrendshistory = pytrends.get_historical_interest([search], year_start=init_date.year, month_start=init_date.month, day_start=init_date.day,year_end=final_date.year, month_end=final_date.month, day_end=final_date.day,  cat=0, geo='BR', gprop='', sleep=2)
google = googletrendshistory.resample('D')[search].mean()