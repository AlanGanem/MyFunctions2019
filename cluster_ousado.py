from func_product_finder import product_finder, product_finder_fasttext
from func_apply_word_embedings import apply_word_embedings
import pandas as pd
import math
import numpy as np
import fastcluster
import scipy.cluster as Cluster
from func_search_engine import  search_engine_fasttext
import collections
import datetime
import scipy.cluster
import mpu
from random import sample
import pickle


class products_db_finder:
    
    def __init__(self):
        
        return None
    
    def __getitem__(self,item):
        return item
    
    
    def fu(self,x, frac = 0.8):
        try:
            d = {}
            d['starting_date'] = max(x['starting_date'])
            d['last_modified_date'] = max(x['last_modified_date'])
            d['category_id'] = sample(mpu.datastructures.flatten(list(x['category_id'])),min(20,math.ceil(len(mpu.datastructures.flatten(list(x['category_id'])))*frac)))
            d['ad_title'] = sample(mpu.datastructures.flatten(list(x['ad_title'])),min(20,math.ceil(len(mpu.datastructures.flatten(list(x['ad_title'])))*frac)))
            d['word_vector'] = self.vector_mean(x['word_vector'],x['counter'])
            d['ad_id'] = int(np.median(np.repeat(x['ad_id'],max(x['counter']))))                       
            try:
                d['counter'] = x['counter'].sum()
            except:
                d['counter'] = len(x)            
        except KeyError:
            
            d = {}
            d['starting_date'] = max(x['starting_date'])
            d['last_modified_date'] = max(x['last_modified_date'])
            d['category_id'] = sample(mpu.datastructures.flatten(list(x['category_id'])),min(20,math.ceil(len(mpu.datastructures.flatten(list(x['category_id'])))*frac)))
            d['ad_title'] = sample(mpu.datastructures.flatten(list(x['ad_title'])),min(20,math.ceil(len(mpu.datastructures.flatten(list(x['ad_title'])))*frac)))
            d['word_vector'] = self.vector_mean(x['word_vector'])
            d['ad_id'] = int(np.median(np.repeat(x['ad_id'],max(x['counter']))))
            try:
                d['counter'] = x['counter'].sum()
            except:
                d['counter'] = len(x)
            
        result = pd.Series(data = d, index=['starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id'])
        return result
    
    def vector_mean(self,y, weights = None):
        word_vectors = [vector[0] for vector in y]
        
        if len(word_vectors) == 1:
            #product_word_vector = [word_vectors[0][0].reshape(1,-1)]
            product_word_vector = word_vectors
        else:
            product_word_vector = [np.average(np.array(word_vectors),axis=0,weights = weights)]
        return product_word_vector

    
    
    def read_csv(self,sql_query_path = r'C:\ProductClustering\productsDB\titles_19_03_2019.csv'):
        self.sql_query_path = sql_query_path
        self.raw_df = pd.read_csv(sql_query_path)
    
    def sampler(self,sample_size = 400,sample_fraction = 0.5, per_category=True):
        self.sample_size = sample_size
        self.sample_fraction = sample_fraction
        titles =  self.raw_df
        titles = titles.dropna()
        titles = titles.assign(ad_title =  titles.ad_title.astype(str))
        if per_category:
            titles_sample = titles.groupby('category_id').apply(lambda x: x.sample(min(sample_size, math.ceil(sample_fraction*len(x))))).reset_index(drop=True)
        else:
            titles_sample = titles.sample(sample_size)
        self.sampled_ads = titles_sample       
        return titles_sample
        
    def graph_communities(self,sample, min_value_=0.6,topn_ = 400, k1=30,expected_density = 0.1):
        graph_clusters = product_finder(sample,min_value_=min_value_, topn_ = topn_ , k1=k1,expected_density =expected_density )
        graph_clusters = apply_word_embedings(graph_clusters['clustered_data']).assign(starting_date = datetime.datetime.today().strftime('%Y-%m-%d'), last_modified_date = datetime.datetime.today().strftime('%Y-%m-%d'))
        self.graph_clusters = graph_clusters.assign(counter = 1)
        return graph_clusters.assign(counter = 1)

    def group_by_product(self,title_clusters, min_elements = 0, prod_id_column ='product_id'):       
        
        try:
            title_clusters.counter
        except AttributeError:
            title_clusters = title_clusters.assign(counter = 1)
            
        title_clusters_joinned = title_clusters.groupby(prod_id_column).apply(self.fu).reset_index(drop = False)
        if len(title_clusters_joinned)>0:    
            
            title_clusters_joinned = title_clusters_joinned[title_clusters_joinned.counter >= min_elements]
            title_clusters_joinned.columns = ['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']
        else:
            title_clusters_joinned = pd.DataFrame(columns = ['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id'])
        return title_clusters_joinned
    
    def linkage(self,title_clusters, method = 'ward'):
        
        try:
            data = np.array([i[0][0] for i in title_clusters.word_vector])
            Z = fastcluster.linkage_vector(data, method = method)
        except AttributeError:
            title_clusters = apply_word_embedings(title_clusters)
            data = np.array([i[0][0] for i in title_clusters.word_vector])
            Z = fastcluster.linkage_vector(data, method = method)
               
        return Z
    
    def hierarchycal_clustering(self,linkage_matrix,threshold =0.3,criterion='distance',depth=2, R=None, monocrit=None):        
        cluster_labels = scipy.cluster.hierarchy.fcluster(linkage_matrix,t = threshold, criterion=criterion, depth=depth, R=R, monocrit=monocrit)
        self.hierarchycal_clusters = cluster_labels
        return cluster_labels
    
    def search_engine(self,raw_df, centroids,threshold = 0.3 ,min_sim = 0.9,prod_id_column ='product_id' ,column_name_db = 'word_vector',column_name_data = 'word_vector',pre_computed_word_vectors=False,min_amount_analogous =3):
        
                
        test = search_engine_fasttext(raw_df,centroids, min_sim = min_sim, column_name_db = column_name_db ,column_name_data = column_name_data ,pre_computed_word_vectors=pre_computed_word_vectors)
        data = test
        try:
            data = data.rename(columns = {'product_id_fasttext':'product_id'})
        except:
            pass
        try:
            test = test.assign(last_modified_date = test.date_max,starting_date = '2000-01-01')
        except:
            test = test.assign(last_modified_date = datetime.datetime.today().strftime('%Y-%m-%d'),starting_date = '2000-01-01')
        try:        
            unknown_products = test[test.product_id_fasttext == -1].assign(starting_date = test[test.product_id_fasttext == -1].date_min)
        except:
            unknown_products = test[test.product_id_fasttext == -1].assign(starting_date = datetime.datetime.today().strftime('%Y-%m-%d'))
        
        self.new_existing_products = test[test.product_id_fasttext != -1].rename(columns={'product_id_fasttext': 'product_id'}).assign(counter = 1)
        self.new_existing_products = self.new_existing_products[['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']]
        
        if len(unknown_products) >= min_amount_analogous:

            last_product_id = max(centroids.product_id)
            unknown_data = np.array([i[0][0] for i in unknown_products.word_vector])
            cluster_ = fastcluster.linkage(unknown_data,method = 'ward')
            cluster_labels = Cluster.hierarchy.fcluster(cluster_,0.2)
            unknown_products = unknown_products.assign(product_id = cluster_labels)
            title_clusters_joinned = self.group_by_product(unknown_products, prod_id_column ='product_id')
            title_clusters_joinned.columns = ['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']
            
            new_products = title_clusters_joinned[title_clusters_joinned.counter >= min_amount_analogous]
            dumped_products = title_clusters_joinned[title_clusters_joinned.counter < min_amount_analogous]
            new_products = new_products.assign(product_id = new_products.product_id.apply(lambda x: x+last_product_id))
            
            self.new_products = self.group_by_product(new_products, prod_id_column = 'product_id')

            self.dumped_products = dumped_products
            
        else:
            new_products = pd.DataFrame(columns = ['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id'])
            self.new_products = new_products
            

        return data
    
    def init_products_db(self,prod_db):
        try:
            prod_db.starting_date
            prod_db.last_modified_date
            self.products_db = prod_db[['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']]
        except:
            self.products_db = prod_db.assign(starting_date = '2000-01-01', last_modified_date = '2000-01-01')[['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']]

    def update_products_db(self,new_products,new_existing_products,prod_db = None):
        
        if not prod_db:
            self.products_db = self.group_by_product(self.products_db.append(new_existing_products))
            self.products_db = self.products_db.append(new_products)
        else:
            self.products_db = self.group_by_product(prod_db.append(new_existing_products))
            self.products_db = prod_db.append(new_products)        
        
        self.prod_db_linkage= self.linkage(self.products_db, method = 'ward')
    
    def zoom_out(self,products_db, linkage_matrix, threshold):
        ''' the threshold when calculating linka with ward represents a measure of variance within the clusters '''
        zoom = self.hierarchycal_clustering(self.prod_db_linkage,threshold = threshold, criterion = 'distance',depth = 'none')
        self.zoomed_out_db = products_db[['product_id']].assign(higher_level = zoom)
    
    
    def export_db_dic(self, path, file_name):
         h = open(os.path.join(os.path.dirname(path), file_name), 'wb')
         pickle.dump(self.produts_db.to_dict(),h)
                
    def drop_old_products(self, threshold = 180):       
        
        temp1 = self.products_db
        date_format = "%Y-%m-%d"
        today = datetime.datetime.strptime(datetime.datetime.today().strftime('%Y-%m-%d'),date_format)
        range_len = range(len(temp1))
        date_diff = []            
        for i in range_len:    
            last_modified_date = datetime.datetime.strptime(temp1.last_modified_date.iloc[i], date_format)
            date_diff.append((today-last_modified_date).days)
        temp1 = temp1.assign(unseen_interval = date_diff)
        temp1 = temp1[temp1.unseen_interval <= threshold]
        return temp1
        
        
        