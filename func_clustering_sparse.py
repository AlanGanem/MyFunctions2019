import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unidecode
import string
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import time
from sparse_dot_topn import awesome_cossim_topn as dot_product
from scipy.sparse import csr_matrix, csc_matrix, csgraph
from func_price_clustering import price_clustering
from sklearn.preprocessing import normalize
import pickle
import os

def clustering_sparse(data_, min_size = 2, max_df_ = 1.01, min_df_ = 0, tagging = True, nwords = 5, cumulative = 0.8, sim1 = 0.6,sim2 = 0.5,metric1 = 'cosine', metric2 = 'jaccard',price = False):
    ######### remove stop words, capitals and accents and assign corpus to df
    titulos = data_['ad_title']
    s = time.time()
    trans_table=str.maketrans(string.punctuation+'0123456789',len(string.punctuation+'0123456789')*' ')
    new_stopwords = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','c','x','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','é','todo','tudo','quase','caixa','gb','mb','mp','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','lançamento','é','todo','tudo','quase','caixa','gb','mb','mp','fiscal','nf','ano','completa','box','descric','leia','descrição','descricao','par','barato','2017','2018','2019','2020','importado','combo','unidades','vezes','promoçao','promoção','promocão','promocao','promoçao','novo','brinde','ft','novidade','novidad','oferta','imperdivel','imperdível','luxo','nova','queima','estoque','original','envio','imediato','off','cxs','mais','barata','barato','demais','demai','liquida','br','pra','ml','oferta','unidad','promoc','promo','tamanho','lt','litro']
    corpus = []
    for i in titulos:    
        titulos_ = i.lower().translate(trans_table).split()
        ps = SnowballStemmer('portuguese')
        titulos_ = [unidecode.unidecode(ps.stem(word)) for word in titulos_ if not word in set(stopwords.words('portuguese')+new_stopwords)]
        titulos_ = ' '.join(titulos_)
        corpus.append(titulos_)
    print (time.time()-s)
    data = data_.copy().assign(ad_title_corpus = corpus)

    del titulos
    del ps
    del titulos_
    ###### get the vocabulary of all the dataset:
    count_vectorizer = CountVectorizer(binary = True )
    cv_matrix = count_vectorizer.fit_transform(np.array(data.ad_title_corpus))
    vocabulary = count_vectorizer.vocabulary_
    jacc_dist = pairwise_jaccard_sparse(cv_matrix)
    clustering = DBSCAN(eps=1-0.68, min_samples=3, metric = 'precomputed').fit(jacc_dist)    
    data = data.assign(product_id = clustering.labels_)
    title_ad_product = data[['ad_title','ad_id','product_id']]
    if price:
        data = price_clustering(data,column = 'product_id')
        segmentation_policy = data[1]
        data= data[0]
        return {'clustered_data':data,'cv_matrix':cv_matrix,'vocabulary':vocabulary,'segmentation_policy':segmentation_policy}
    else:
        return {'clustered_data':data,'cv_matrix':cv_matrix,'vocabulary':vocabulary}
        
    
    
def pairwise_cosine_sparse(csr, topn=2000, min_value = 0.1):
    """Computes the cosine distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """   
    csr = csr_matrix(csr).astype(bool,copy=False).astype(int,copy=False)
    csr = csr.astype('float64',copy=False)
    csr = normalize(csr, norm='l2', axis=1)
    intrsct = dot_product(csr,csr.T, topn, min_value)
    intrsct.data[intrsct.data>=1] = 1
    intrsct.data = 1 - intrsct.data
    
    return intrsct
    
    
    
def pairwise_jaccard_sparse(csr, topn=2000, min_value = 2):
    """Computes the Jaccard distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """   
    csr = csr_matrix(csr).astype(bool,copy=False).astype(int,copy=False)
    csr = csr.astype('float64',copy=False)
    csr_rownnz = csr.getnnz(axis=1)
    intrsct = dot_product(csr,csr.T, topn, min_value-0.1)

    nnz_i = np.repeat(csr_rownnz, intrsct.getnnz(axis=1))
    unions = nnz_i + csr_rownnz[intrsct.indices] - intrsct.data
    intrsct.data = 1.0 - intrsct.data / unions
       
    return intrsct




g = os.path.join(os.path.dirname(r'C:\ProductClustering\output_data\\'), 'cv_matrix')
cv_matrix = pickle.load(open(g, 'rb'))
data_janeiro = pd.read_csv(r'C:\Users\PC10\Desktop\Alan\product_clustering\dados\Historical data\janeiro_2019\mes.csv')
g = os.path.join(os.path.dirname(r'C:\ProductClustering\output_data\\'), 'cat_dic')
cat_dic = pickle.load(open(g, 'rb'))

g = os.path.join(os.path.dirname(r'C:\ProductClustering\output_data\\'), 'jacc_dist')
jacc_dist = pickle.load(open(g, 'rb'))

cosine_dist = pairwise_cosine_sparse(cv_matrix, min_value = 0.4)
jacc_dist = pairwise_jaccard_sparse(cv_matrix)

import hdbscan
from collections import Counter

clusteringH = hdbscan.HDBSCAN(min_cluster_size = 3,metric = 'precomputed')

clustering = DBSCAN(eps=1-0.9, min_samples=5, metric = 'precomputed').fit(cosine_dist)
len(set(Counter(clustering.labels_)))
label=Counter(clustering.labels_).most_common()[1][0]
Counter(clustering.labels_).most_common()[1]

i, = np.where(clustering.labels_==label)
data_janeiro.iloc[i]

category_id = '1430'
data_janeiro_teste = data_janeiro[data_janeiro.category_id.isin(cat_dic[category_id])]
cv_teste = cv_matrix[data_janeiro_teste.index.values]
cosine_dist_teste = pairwise_cosine_sparse(cv_teste, min_value = 0.6)
a= csgraph.connected_components(cosine_dist_teste)[1]
i, = np.where(a==0)

cv_teste = cv_teste[i]
cosine_dist_teste = pairwise_cosine_sparse(cv_teste, min_value = 0.6)
a= csgraph.connected_components(cosine_dist_teste)[1]
i, = np.where(a!=0)


clusteringH = hdbscan.HDBSCAN(min_cluster_size = 3,metric = 'precomputed')
clusteringH.fit(cosine_dist_teste)
data_janeiro_teste.iloc[i]
