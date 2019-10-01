
import pandas as pd
import numpy as np
import time
from func_gini import gini
import pickle
import os
from func_get_father_categories import get_father_categories
import tqdm
import urllib.request
import json


def tratamento(data, cond_cat_revenues = '>0', column= 'product_id', min_seller_sales = 0,min_analogous = 1, update_cats =True):    
    ########################## importando módulos ################################       
    print('importing modules;')    
    
    data = data.dropna()
    
    if update_cats:
        cat_dic = get_father_categories()
    else:
        f = open(os.path.join(os.path.dirname(r'C:\ProductClustering\input_Data\\'), 'cat_dic'),'rb')
        cat_dic = pickle.load(f)

    def load_url(url, timeout):
           with urllib.request.urlopen(url, timeout=timeout) as conn:
               return conn.read()
    try:
        r = load_url('https://api.mercadolibre.com/sites/MLB/categories', 10)
        data_r = json.loads(r)
        category_names= {int(i['id'][3:]):i['name'] for i in data_r}
    except:
        category_names = {1000: 'Eletrônicos, Áudio e Vídeo',
 1039: 'Câmeras e Acessórios',
 1051: 'Celulares e Telefones',
 1071: 'Animais',
 1132: 'Brinquedos e Hobbies',
 1144: 'Games',
 1168: 'Música',
 1182: 'Instrumentos Musicais',
 1196: 'Livros',
 1246: 'Beleza e Cuidado Pessoal',
 1276: 'Esportes e Fitness',
 1367: 'Antiguidades',
 1368: 'Arte e Artesanato',
 1384: 'Bebês',
 1403: 'Alimentos e Bebidas',
 1430: 'Calçados, Roupas e Bolsas',
 1459: 'Imóveis',
 1499: 'Agro, Indústria e Comércio',
 1540: 'Serviços',
 1574: 'Casa, Móveis e Decoração',
 1648: 'Informática',
 1743: 'Carros, Motos e Outros',
 1798: 'Coleções e Comics',
 1953: 'Mais Categorias',
 218519: 'Ingressos',
 263532: 'Ferramentas e Construção',
 264586: 'Saúde',
 3281: 'Filmes e Seriados',
 3937: 'Joias e Relógios',
 5672: 'Acessórios para Veículos',
 5726: 'Eletrodomésticos'}    
            


    print('applying initial filters to data;\n')
    ############################  interface usuário ###############################
    #data_not_paused = data[data.interval == max(data.interval)]
    #data_paused = data[data.interval != max(data.interval)]
    ###############################################################################
    print('filters applyed:\n' + 'listed by '+column+'\n'+'considering sellers with salles greater or equals to ' + str(min_seller_sales))
    print('creating lists by ' + column + ';\n' )   
    ####### rotina que retorna lista de dataframes por "column" que satisfazem "cond" #########################################    
    data = data.replace(to_replace=0.1, value=0)
    daily_sold_difference = data['sold_difference']/(data['active_interval'])
    data = data.assign(daily_sold_difference = daily_sold_difference)
    data = data.assign(daily_revenues = daily_sold_difference*data.price_min)
    print('calculating and appending new metrics;')
    
    product_id_by_price_=[]
   # title_corpus_=[]
    product_name_=[]
    amount_of_analogous_=[]
    product_id_=[]
    category_sold_=[]
    amount_of_sellers_=[]
    relevance_50_=[]
    activity_ratio_median_=[]
    sold_ratio_min_=[]
    sold_ratio_25_=[]
    sold_ratio_50_=[]
    sold_ratio_75_=[]
    sold_ratio_max_=[]
    daily_revenues_min_=[]
    daily_revenues_25_=[]
    daily_revenues_50_=[]
    daily_revenues_75_=[]
    daily_revenues_max_=[]
    daily_sold_min_=[]
    daily_sold_25_=[]
    daily_sold_50_=[]
    daily_sold_75_=[]
    daily_sold_max_=[]
    top_sellers_=[]
    relative_price_range_=[] 
    price_range_=[]
    top_ads_=[]
    #product_word_vector_=[]
    activity_ratio_top_1_=[]
    activity_ratio_top_2_=[]
    activity_ratio_top_3_=[]
    product_views_=[]
    product_conversion_ratio_=[]
    category_revenues_= []
    product_sold_ratio_ = []
    product_views_ratio_ = []
    gini_coefficient_revenue_ =[]
    father_category_ =[]
    min_price_ = []
    max_price_ = []
    ad_id_median_ =[]
    revenues_by_supply_ = []
    k=0
    j=0
    list_of_labels = list(set(data[column]))
    amount_of_products = len(list(set(data[column])))
    percent = int(amount_of_products/100)
    percent_multiples =[percent*n for n in range(0,100)]         
    time_forecast = int(amount_of_products*0.008)
    for i in tqdm.tqdm(list_of_labels):        
        if k == 0:
            s = time.time()
            
        by_column_dic_i = data[data[column] == i]
        if len(by_column_dic_i) >= min_analogous:
            product_id_i = by_column_dic_i.product_id.max()
            try:
                product_id_by_price_i = by_column_dic_i.product_id_by_price.max()
            except:
                product_id_by_price_i = -1
            category_revenues = by_column_dic_i['daily_revenues'].sum()
            len_by_column_dic_i=len(by_column_dic_i)
            sellers_matching_condition = pd.unique(by_column_dic_i.seller_id[by_column_dic_i.sold_difference >= min_seller_sales])
            total_ads = len_by_column_dic_i
            by_column_dic_i_groupby_seller_id = by_column_dic_i.groupby('seller_id')        
            by_column_dic_i_groupby_seller_id_revenues = by_column_dic_i_groupby_seller_id.period_revenues.sum()
            gini_revenues_list = [i for i in by_column_dic_i_groupby_seller_id_revenues]
            gini_revenues_list.sort()
            gini_coefficient_revenue = gini(np.array(gini_revenues_list))       
            sellers_sold_difference_sum = by_column_dic_i_groupby_seller_id.sold_difference.sum()
            sellers_sold_quantity_sum = by_column_dic_i_groupby_seller_id.sold_quantity_max.sum()
            sold_ratio_describe = (sellers_sold_difference_sum/sellers_sold_quantity_sum).describe()            
            revenues_describe = by_column_dic_i_groupby_seller_id.daily_revenues.sum().describe()
            activity_ratio_median = by_column_dic_i.activity_ratio.median()
            daily_sold_describe = by_column_dic_i_groupby_seller_id.daily_sold_difference.sum().describe()
            min_cat_price = min(by_column_dic_i['price_min'])
            max_cat_price = max(by_column_dic_i['price_min'])
            price_range = '%s - %s' % (min_cat_price,max_cat_price)
            amount_of_analogous=len(by_column_dic_i)
            amount_of_sellers = len(sellers_matching_condition)
            relevance_50 = (by_column_dic_i_groupby_seller_id.median_position.max().describe())['50%']
            category_sold = by_column_dic_i['sold_difference'].sum()
            sold_ratio_min = sold_ratio_describe['min']
            sold_ratio_25 = sold_ratio_describe['25%']
            sold_ratio_50 = sold_ratio_describe['50%']
            sold_ratio_75 = sold_ratio_describe['75%']
            sold_ratio_max = sold_ratio_describe['max']
            product_sold_ratio = by_column_dic_i.sold_difference.sum()/by_column_dic_i.sold_quantity_max.sum()
            product_views_ratio = by_column_dic_i.period_views.sum()/by_column_dic_i.period_views_max.sum()
            daily_revenues_min = revenues_describe['min']
            daily_revenues_25 = revenues_describe['25%']
            daily_revenues_50 = revenues_describe['50%']
            daily_revenues_75 = revenues_describe['75%']
            daily_revenues_max = revenues_describe['max']
            daily_sold_min = daily_sold_describe['min']
            daily_sold_25 = daily_sold_describe['25%']
            daily_sold_50 = daily_sold_describe['50%']
            daily_sold_75 = daily_sold_describe['75%']
            daily_sold_max = daily_sold_describe['max']
            revenues_by_supply = by_column_dic_i['period_revenues'].max()/amount_of_sellers
            top_sellers = by_column_dic_i_groupby_seller_id.sold_difference.sum().sort_values(ascending = False)[0:3].index.values
            top_ads = list(set([by_column_dic_i.nlargest(1,'sold_difference')['ad_id'].max(),by_column_dic_i.nlargest(2,'sold_difference')['ad_id'].max(),by_column_dic_i.nlargest(3,'sold_difference')['ad_id'].max()]))
            relative_price_range = abs(min_cat_price-max_cat_price)/max(1,min_cat_price)
            #word_vectors = [vector[0] for vector in by_column_dic_i['word_vector']]
            #if len(word_vectors) == 1:
                #product_word_vector = [word_vectors[0][0].reshape(1,-1)]
            #    product_word_vector = word_vectors
            #else:
            #    product_word_vector = [np.average(np.array(word_vectors),axis=0)]
            product_views = by_column_dic_i.period_views.sum()
            product_conversion_ratio = by_column_dic_i.sold_difference.sum()/max(1,product_views)
            l = 1
            m = top_sellers
            activity_ratio_top_1 = by_column_dic_i[by_column_dic_i.seller_id == m[0]].activity_ratio.mean()
            try:
                activity_ratio_top_2 = by_column_dic_i[by_column_dic_i.seller_id == m[1]].activity_ratio.mean()
            except IndexError:
                activity_ratio_top_2 = 0
            try:
                activity_ratio_top_3 = by_column_dic_i[by_column_dic_i.seller_id == m[2]].activity_ratio.mean()
            except IndexError:
                activity_ratio_top_3 = 0
            
            value = by_column_dic_i.category_id.mode().values[0]
            try:
                father_category = category_names[int(cat_dic[str(value)])]
            except IndexError:
                father_category = 'not_found'
            except KeyError:
                father_category = 'not_found'
            
            ad_id_median = int(by_column_dic_i.ad_id.astype(float).median())
            
            ###############appending new values
            #title_corpus_.append(by_column_dic_i.ad_title_corpus.mode()[0])
            product_name_.append(by_column_dic_i.ad_title[by_column_dic_i.ad_id == top_ads[0]].max())
            amount_of_analogous_.append(amount_of_analogous)
            product_id_.append(product_id_i)
            product_id_by_price_.append(product_id_by_price_i)
            father_category_.append(father_category)
            category_sold_.append(category_sold)
            category_revenues_.append(category_revenues)
            amount_of_sellers_.append(amount_of_sellers)
            relevance_50_.append(relevance_50)
            activity_ratio_median_.append(activity_ratio_median)
            sold_ratio_min_.append(sold_ratio_min)
            sold_ratio_25_.append(sold_ratio_25)
            sold_ratio_50_.append(sold_ratio_50)
            sold_ratio_75_.append(sold_ratio_75)
            sold_ratio_max_.append(sold_ratio_max)
            daily_revenues_min_.append(daily_revenues_min)
            daily_revenues_25_.append(daily_revenues_25)
            daily_revenues_50_.append(daily_revenues_50)
            daily_revenues_75_.append(daily_revenues_75)
            daily_revenues_max_.append(daily_revenues_max)
            daily_sold_min_.append(daily_sold_min)
            daily_sold_25_.append(daily_sold_25)
            daily_sold_50_.append(daily_sold_50)
            daily_sold_75_.append(daily_sold_75)
            daily_sold_max_.append(daily_sold_max)
            top_sellers_.append(top_sellers)
            relative_price_range_.append(relative_price_range)
            min_price_.append(min_cat_price)
            max_price_.append(max_cat_price)
            top_ads_.append(top_ads)
            #product_word_vector_.append(product_word_vector)
            activity_ratio_top_1_.append(activity_ratio_top_1)
            activity_ratio_top_2_.append(activity_ratio_top_2)
            activity_ratio_top_3_.append(activity_ratio_top_3)
            product_views_.append(product_views)
            product_conversion_ratio_.append(product_conversion_ratio)
            product_sold_ratio_.append(product_sold_ratio)
            product_views_ratio_.append(product_views_ratio)
            gini_coefficient_revenue_.append(gini_coefficient_revenue)
            ad_id_median_.append(ad_id_median)
            revenues_by_supply_.append(revenues_by_supply)
        else:
            pass
        
        k+=1
        #if k == time_forecast:
        #    print('/n estimated duration:'+str(int((time.time()-s)*amount_of_products/60/time_forecast))+' minutes')
        #if k%percent == 0:
        #    print(str(int(k/percent))+'%',end="", flush=True)
        #if k in percent_mutiples:
        #    print (str(k/amount_of_products*100)+'%')
    ################################################################################
    ranking = pd.DataFrame().assign(
    #title_corpus=title_corpus_,
    product_name=product_name_,
    amount_of_analogous=amount_of_analogous_,
    product_id=product_id_,
    product_id_by_price = product_id_by_price_,
    father_category = father_category_,
    product_sold=list(map(float,category_sold_)),
    product_revenues=  category_revenues_,
    amount_of_sellers=amount_of_sellers_,
    relevance_50=relevance_50_,
    activity_ratio_median=activity_ratio_median_,
    sold_ratio_min=sold_ratio_min_,
    sold_ratio_25=sold_ratio_25_,
    sold_ratio_50=sold_ratio_50_,
    sold_ratio_75=sold_ratio_75_,
    sold_ratio_max=sold_ratio_max_,
    daily_revenues_min=daily_revenues_min_,
    daily_revenues_25=daily_revenues_25_,
    daily_revenues_50=daily_revenues_50_,
    daily_revenues_75=daily_revenues_75_,
    daily_revenues_max=daily_revenues_max_,
    daily_sold_min=daily_sold_min_,
    daily_sold_25=daily_sold_25_,
    daily_sold_50=daily_sold_50_,
    daily_sold_75=daily_sold_75_,
    daily_sold_max=daily_sold_max_,
    gini_coefficient_revenue = gini_coefficient_revenue_,
    top_sellers=top_sellers_,
    relative_price_range=relative_price_range_,
    product_min_price= list(map(float,min_price_)),
    product_max_price =  list(map(float,max_price_)),
    top_ads=top_ads_,
    #product_word_vector=product_word_vector_,
    activity_ratio_top_1=list(map(float,activity_ratio_top_1_)),
    activity_ratio_top_2=list(map(float,activity_ratio_top_2_)),
    activity_ratio_top_3=list(map(float,activity_ratio_top_3_)),
    product_views=product_views_,
    product_conversion_ratio=product_conversion_ratio_,
    product_sold_ratio = product_sold_ratio_,
    product_views_ratio = product_sold_ratio_,
    ad_id_median = ad_id_median_,
    revenues_by_supply = revenues_by_supply_)
    print('done\n')
    
    return ranking
