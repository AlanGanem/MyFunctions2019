from func_product_finder import product_finder, product_finder_fasttext
from func_apply_word_embedings import apply_word_embedings
import pandas as pd
import math
import numpy as np
import fastcluster
import scipy.cluster as Cluster
from func_search_engine import  search_engine_fasttext, pairwise_cosine_sparse_sim
import collections
import datetime
import scipy.cluster
import mpu
from random import sample
import pickle
import os
from scipy.sparse import csr_matrix

class products_db_finder:
    
    def __init__(self, model_name = 'model_fast_text_sg_40'):
        self.model_name = model_name
        return None
    
    def __getitem__(self,item):
        return item

    def fu(self,x, frac = 0.8):
        
        if set(['starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']) - set(x.columns) == set():
            if len(x)>1:
                d = {}        
                
                #x['starting_date'] = x['starting_date'].astype(str)
                d['starting_date'] = min(x['starting_date'].astype(str))
                d['last_modified_date'] = max(x['last_modified_date'].astype(str))
                d['category_id'] = mpu.datastructures.flatten(list(set(mpu.datastructures.flatten(list(x['category_id'])))))
                if len(x['ad_title'])>1:
                    d['ad_title'] = sample(mpu.datastructures.flatten(list(x['ad_title'])),min(20,math.ceil(len(mpu.datastructures.flatten(list(x['ad_title'])))*frac)))
                else:
                    d['ad_title'] = mpu.datastructures.flatten([x['ad_title'].max()])
                d['word_vector'] = self.vector_mean(x['word_vector'],x['counter'])
                d['ad_id'] = list(set(mpu.datastructures.flatten(list(x['ad_id']))))
                try:
                    d['counter'] = len(list(set(mpu.datastructures.flatten(list(set(x['ad_id']))))))
                except:
                    d['counter'] = len(x)
                    
                result = pd.Series(data = d, index=['starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id'])
                return result
            else:
                return x[['starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']].max()
        
        else:
            d = {}        
                
            #x['starting_date'] = x['starting_date'].astype(str)
            d['starting_date'] = min(x['starting_date'].astype(str))
            d['last_modified_date'] = max(x['last_modified_date'].astype(str))
            d['category_id'] = mpu.datastructures.flatten(list(set(mpu.datastructures.flatten(list(x['category_id'])))))
            if len(x['ad_title'])>1:
                d['ad_title'] = sample(mpu.datastructures.flatten(list(x['ad_title'])),min(20,math.ceil(len(mpu.datastructures.flatten(list(x['ad_title'])))*frac)))
            else:
                d['ad_title'] = mpu.datastructures.flatten([x['ad_title'].max()])
            d['word_vector'] = self.vector_mean(x['word_vector'],x['counter'])
            d['ad_id'] = list(set(mpu.datastructures.flatten(list(x['ad_id']))))
            try:
                d['counter'] = len(list(set(mpu.datastructures.flatten(list(set(x['ad_id']))))))
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
    
    def read_csv(self,sql_query_path = r'C:\ProductClustering\productsDB\titles_19_03_2019.csv', pandas_df = None):
        if pandas_df is None:
            self.sql_query_path = sql_query_path
            self.raw_df = pd.read_csv(sql_query_path)
        else:
            self.sql_query_path = 'precomputed'
            self.raw_df = pandas_df
    def sampler(self,sample_size = 400,sample_fraction = 0.5, per_category=True, pandas_df = None):
        
        if not pandas_df is None:
            self.sampled_ads = pandas_df
            return pandas_df
        
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
        
    def graph_communities(self,data, min_value_=0.6,topn_ = 400, k1=30,expected_density = 0.1,graph_communities_df = None):
        if  not graph_communities_df is None:
            self.graph_clusters = graph_communities_df
            return graph_communities_df
        graph_clusters = product_finder (data, min_value_=min_value_, topn_ = topn_ , k1=k1,expected_density =expected_density )
        try:
            graph_clusters = apply_word_embedings(graph_clusters['clustered_data'], model_name = self.model_name).assign(starting_date = graph_clusters['clustered_data'].date_min, last_modified_date = graph_clusters['clustered_data'].date_max)
        except:
            graph_clusters = apply_word_embedings(graph_clusters['clustered_data'], model_name = self.model_name).assign(starting_date = datetime.datetime.today().strftime('%Y-%m-%d'), last_modified_date = datetime.datetime.today().strftime('%Y-%m-%d'))
        if not 'counter' in graph_clusters.columns:
            self.graph_clusters = graph_clusters.assign(counter = 1)
        else:
            self.graph_clusters = graph_clusters
        return self.graph_clusters

    def group_by_product(self,title_clusters, min_elements = 0, prod_id_column ='product_id'):
        
        try:
            title_clusters['counter']
        except AttributeError:
            title_clusters = title_clusters.assign(counter = 1)
        
        
        title_clusters_joinned = title_clusters.groupby(prod_id_column).apply(self.fu).reset_index(drop = False)
        print(title_clusters_joinned)
        if len(title_clusters_joinned)>0:    
            
            title_clusters_joinned = title_clusters_joinned[title_clusters_joinned['counter'] >= min_elements]
            title_clusters_joinned.columns = ['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']
        else:
            title_clusters_joinned = pd.DataFrame(columns = ['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id'])
        return title_clusters_joinned.set_index(np.arange(len(title_clusters_joinned)))
    
    def linkage(self,title_clusters, method = 'ward', linkage_matrix = None):
        if not linkage_matrix is None:
            self.linkage_matrix = linkage_matrix
            return linkage_matrix
        
        try:
            data = np.array([i[0][0] for i in title_clusters.word_vector])
            Z = fastcluster.linkage_vector(data, method = method)
        except AttributeError:
            title_clusters = apply_word_embedings(title_clusters, model_name = self.model_name)
            data = np.array([i[0][0] for i in title_clusters.word_vector])
            Z = fastcluster.linkage_vector(data, method = method)
               
        return Z
    
    def hierarchycal_clustering(self,linkage_matrix,threshold =0.5,criterion='distance',depth=2, R=None, monocrit=None):        
        
        cluster_labels = scipy.cluster.hierarchy.fcluster(linkage_matrix,t = threshold, criterion=criterion, depth=depth, R=R, monocrit=monocrit)
        self.hierarchycal_clusters = cluster_labels
        return cluster_labels
    
    def search_engine(self,raw_df, centroids,threshold = 0.5 ,min_sim = 0.9,model_name = 'model_fast_text_sg_40',prod_id_column ='product_id' ,column_name_db = 'word_vector',column_name_data = 'word_vector',pre_computed_word_vectors=False,min_amount_analogous =3,clustering_algorithm = 'agglomerative'):
        '''
        performs a serach for  similarity of word vector from a dataframe in a precalculated reference DB
        
        raw_df is the unlabeled data
        centroids  is the reference DB
        threshold is the hierarchichal clustering threshold distance
        min sim is the minimum similarity in order to assign an ad_title to a prodcut_id tag
        
        '''
        
        last_product_id = max(centroids.product_id)
           
        if not ('category_id' in raw_df.columns):
            raw_df = raw_df.assign(category_id = 0)          
        
        test = search_engine_fasttext(raw_df,centroids, min_sim = min_sim, model_name = model_name,column_name_db = column_name_db ,column_name_data = column_name_data ,pre_computed_word_vectors=pre_computed_word_vectors)
        test = test.rename(columns = {'product_id_fasttext':'product_id'})
        test = test[['date_min','date_max','category_id','ad_title','word_vector','ad_id','product_id']]
        data = test
        print('{} unlabeled ads'.format(len(test[test.product_id == -1]['product_id'])))

        try:
            test = test.assign(last_modified_date = test.date_max,starting_date = test.date_min)
        except:
            test = test.assign(last_modified_date = datetime.datetime.today().strftime('%Y-%m-%d'),starting_date = '2000-01-01')
        try:        
            unknown_products = test[test.product_id == -1].assign(starting_date = test[test.product_id == -1].date_min)
        except:
            print(test.product_id)
            unknown_products = test[test.product_id == -1].assign(starting_date = datetime.datetime.today().strftime('%Y-%m-%d'))
        
        try:
            test = test.drop('date_max', axis = 1)
        except:
            pass
        try:
            test = test.drop('date_min', axis = 1)
        except:
            pass
        try:
            unknown_products = unknown_products.drop('date_max', axis = 1)
        except:
            pass
        try:
            unknown_products = unknown_products.drop('date_min', axis = 1)
        except:
            pass
            
        
        print('groupying new products')         
             
        self.new_existing_products = test[test.product_id != -1].rename(columns={'product_id': 'product_id'}).assign(counter = 1)
        self.new_existing_products = self.group_by_product(self.new_existing_products[['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']])
        
        if len(unknown_products) >= min_amount_analogous:            
            
            if clustering_algorithm == 'agglomerative':
                unknown_data = np.array([i[0][0] for i in unknown_products.word_vector])
                cluster_ = fastcluster.linkage_vector(unknown_data,method = 'ward')
                cluster_labels = Cluster.hierarchy.fcluster(cluster_,threshold)
                unknown_products = unknown_products.assign(product_id = cluster_labels)
                
            elif clustering_algorithm == 'community':
                unknown_products = self.graph_communities(unknown_products , min_value_=0.8,topn_ = 400, k1=50,expected_density = 0.1,graph_communities_df = None)
            
            
            title_clusters_joinned = self.group_by_product(unknown_products, prod_id_column ='product_id')
            new_products = title_clusters_joinned[title_clusters_joinned.counter >= min_amount_analogous]
            dumped_products = title_clusters_joinned[title_clusters_joinned.counter < min_amount_analogous]
            new_products = new_products.assign(product_id = new_products.product_id.apply(lambda x: x+last_product_id+1))
            
            self.new_products = new_products
            self.dumped_products = dumped_products
            
            print(title_clusters_joinned)
            
        else:
            new_products = pd.DataFrame(columns = ['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id'])
            self.new_products = new_products
        try:    
            print(str(len(self.new_existing_products)) + ' products that already exist in data base')
        except:
            pass
        try:
            print(str(len(self.new_products)) + ' new products found')
        except:
            pass
        try:
            print(str(len(self.dumped_products)) + ' ads dumped')
        except:
            pass
        
        self.new_existing_products = self.new_existing_products.set_index(np.arange(len(self.new_existing_products))) 
        self.new_products= self.new_products.set_index(np.arange(len(self.new_products))) 
        
        try:
            self.dumped_products = self.dumped_products.set_index(np.arange(len(self.dumped_products))) 
        except:
            try:
                self.dumped_products = dumped_products.set_index(np.arange(len(dumped_products)))     
            except:
                pass
        
        return data
    
    def handle_unlabeled(self,data,max_product_id, clustering_algorithm = 'agglomerative'):
        unknown_products = apply_word_embedings(data, model_name = self.model_name)
        
        if clustering_algorithm == 'agglomerative':
                
                unknown_data = np.array([i[0][0] for i in unknown_products.word_vector])
                cluster_ = fastcluster.linkage_vector(unknown_data,method = 'ward')
                cluster_labels = Cluster.hierarchy.fcluster(cluster_,0.2)
                unknown_products = unknown_products.assign(product_id = cluster_labels)
                
        elif clustering_algorithm == 'community':
                unknown_products = self.graph_communities(unknown_products , min_value_=0.8,topn_ = 400, k1=50,expected_density = 0.1,graph_communities_df = None)
            
    
    def init_products_db(self,prod_db):
        try:
            prod_db.starting_date
            prod_db.last_modified_date
            self.products_db = prod_db[['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']]
        except:
            self.products_db = prod_db.assign(starting_date = '2000-01-01', last_modified_date = '2000-01-01')[['product_id','starting_date','last_modified_date','category_id','ad_title','counter','word_vector','ad_id']]
        
        self.products_db = self.products_db.set_index(np.arange(len(self.products_db)))
        self.last_product_id= self.products_db.product_id.max()

    def update_products_db(self,new_products,new_existing_products, update_linkage = False):
        
        if len(self.new_products) > 0:
            new_products= new_products.assign(product_id = new_products.product_id + self.products_db.product_id.max())
        
        if len(self.new_products) > 0:
            assert self.products_db.product_id.max() < new_products.product_id.min() 
        
        for i in new_products:
            assert i in self.products_db.columns
        for i in new_existing_products:
            assert i in self.products_db.columns
        
        if len(new_existing_products) >0:
            self.products_db = self.group_by_product(self.products_db.append(new_existing_products))
        
        self.products_db = self.products_db.append(new_products)
        
        if  update_linkage:
            self.linkage_matrix = self.linkage(self.products_db, method = 'ward')
            
        self.last_product_id= max(self.products_db.product_id)
        self.products_db = self.products_db.set_index(np.arange(len(self.products_db)))
        print('products data base successfully updated')
        
    def zoom_out(self,products_db, linkage_matrix, threshold):
        ''' the threshold when calculating linka with ward represents a measure of variance within the clusters '''
        zoom = self.hierarchycal_clustering(self.linkage_matrix,threshold = threshold, criterion = 'distance',depth = 'none')
        self.zoomed_out_db = products_db[['product_id']].assign(higher_level = zoom)
    
    
    def export_db_dic(self, path, file_name):
         
        h = open(os.path.join(os.path.dirname(path), file_name), 'wb')
        pickle.dump(self.products_db.to_dict(),h)
         
    def import_db_dic(self, path, file_name):
        
       g = os.path.join(os.path.dirname(path), file_name)
       self.products_db_dic = pickle.load(open(g, 'rb'))
        
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

    def get_similar_products(self, product_id= 123, top_n = 10,title = None , column_name = 'word_vector'):
        
        if type(title) == None:
            try:
                csr1 = csr_matrix(np.array([i[0][0] for i in self.products_db[column_name][self.products_db['product_id'].isin(product_id)]]))
                ads = product_id
            except:
                csr1 = csr_matrix(np.array([i[0][0] for i in self.products_db[column_name][self.products_db['product_id'].isin([product_id])]]))
                ads = [product_id]
        else:
            if type(title) == str:
                title = [title]
            titles = apply_word_embedings(pd.DataFrame({'ad_title': title}), model_name = self.model_name)
            csr1 = csr_matrix(np.array([i[0][0] for i in titles['word_vector']]))
            ads = title
            
        csr2 = csr_matrix(np.array([i[0][0] for i in self.products_db[column_name]]))
        sim_matrix = pairwise_cosine_sparse_sim(csr1,csr2, topn=top_n, min_value = 0, word_embedings = True)
        

        labels = {}
        for index, ad in enumerate(ads):    
            labels[ad] = self.products_db.iloc[np.where(sim_matrix[index].A > 0.01)[1]].assign(similarity = sim_matrix[index].A[np.where(sim_matrix[index].A > 0.01)]).sort_values(by = 'similarity', ascending = False)
        
        return labels
    
    def merge_products(self, data, merge_dic, mode  = 'by_id',update_db= False):
        avalible_modes = ['by_id', 'by_similarity']
        if not (mode in avalible_modes):
            raise Exception('mode must be one of ' + str(avalible_modes))
        if mode == 'by_similarity':
            print('mode not yet implemented')
            return
        if mode =='by_id':
            for key in merge_dic.keys():                
                    merged_products = data[data['product_id'].isin((merge_dic[key]+[key]))].assign(product_id = max(key, max(merge_dic[key])))                    
                    data = data.drop(merged_products.index.values)
                    data = data.append(self.group_by_product(merged_products),verify_integrity = True)
        if update_db:
            self.products_db = data
            
        return data
