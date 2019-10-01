from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unidecode
import string
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sparse_dot_topn import awesome_cossim_topn as dot_product
import datetime
import time
from igraph import *
import scipy
import fastcluster

def product_finder(data_, k1= 30,topn_=4000, min_value_ = 0.5,expected_density = 1):   
    
    
    datetime.datetime.today().strftime('%Y-%m-%d')
######### remove stop words, capitals and accents and assign corpus to df
    titulos = data_['ad_title']
    s = time.time()
    trans_table=str.maketrans(string.punctuation+'0123456789',len(string.punctuation+'0123456789')*' ')
    new_stopwords = ['1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','c','x','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','é','todo','tudo','quase','caixa','gb','mb','mp','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','lançamento','é','todo','tudo','quase','caixa','gb','mb','mp','fiscal','nf','ano','completa','box','descric','leia','descrição','descricao','par','barato','2017','2018','2019','2020','importado','combo','unidades','vezes','promoçao','promoção','promocão','promocao','promoçao','novo','brinde','ft','novidade','novidad','oferta','imperdivel','imperdível','luxo','nova','queima','estoque','original','envio','imediato','off','cxs','mais','barata','barato','demais','demai','liquida','br','pra','ml','oferta','unidad','promoc','promo','tamanho','lt','litro']    
    stopwords_ = list(set(stopwords.words('portuguese')+new_stopwords))
    ps = SnowballStemmer('portuguese')
    corpus = []
    for i in titulos:    
        titulos_ = i.lower().translate(trans_table).split()
        titulos_ = [unidecode.unidecode(ps.stem(word)) for word in titulos_ if not word in stopwords_]
        titulos_ = ' '.join(titulos_)
        corpus.append(titulos_)
    print (str(time.time()-s)+'s for text preprocessing')
    data = data_.copy().assign(ad_title_corpus = corpus)

    del titulos
    del ps
    del titulos_
    ###### get the vocabulary of all the dataset:
    s = time.time()
    print('creating matrix of vector')
    count_vectorizer = CountVectorizer(binary = True )
    print('creating  cv_matrix')
    cv_matrix = count_vectorizer.fit_transform(np.array(data.ad_title_corpus))
    vocabulary = count_vectorizer.vocabulary_   
    print('calculating similarity matrix')
    cosine_sim = pairwise_cosine_sparse_sim(cv_matrix, topn = topn_, min_value=min_value_,expected_density = expected_density)    
    print (str(time.time()-s)+'s for cosine similarity computing')
    s = time.time()
    print('generating similarities graph')
    sources, targets = cosine_sim.nonzero()
    g = Graph(list(zip(sources.tolist(), targets.tolist())))
    print (str(time.time()-s)+'s for graph generation')
    s= time.time()
    print('creating graph communities')
    clusters= g.community_multilevel(weights = np.exp(k1*cosine_sim.data))
    data=data.assign(product_id = clusters.membership)
    try:
        title_ad_product = data[['ad_title','ad_id','product_id']]
    except:
        title_ad_product = None
    print (str(time.time()-s)+'s for clusters computation')
    
    #clusters_graph = clusters.cluster_graph()
    #clusters_graph.transitivity_avglocal_undirected()
    #clusters.modularity
    
    
    #Counter(clusters.membership).most_common()[0]
    #clusters= g.community_label_propagation(weights = np.exp(k1*cosine_sim.data))
    
    
    #clusters.modularity
    
    #cter = Counter(clusters.membership)
    #cterkeys = Counter(clusters.membership).keys()
    #ones = [key  for key in cterkeys if cter[key] == 1]
    #len(ones)/len(set(clusters.membership))
    return {'clustered_data':data ,'cv_matrix':cv_matrix,'vocabulary':vocabulary, 'title_ad_product':title_ad_product, 'sim_matrix_density': cosine_sim.size/(cosine_sim.shape[0]*cosine_sim.shape[1])}

def product_finder_fasttext(data_, k1= 30,topn_=4000, min_value_ = 0.95,expected_density = 1.1, sparse=False, clustering_algorithm = 'agglomerative'):   
    
    if clustering_algorithm == 'community':
        print('creating  cv_matrix')
        data = [i[0][0] for i in data_['word_vector']]
        cv_matrix = csr_matrix(np.array(data))
        print('calculating similarity matrix')
        s = time.time()
        cosine_sim = pairwise_cosine_sparse_sim(cv_matrix, topn = topn_, min_value=min_value_,expected_density = expected_density, sparse= False)
        print (str(time.time()-s)+'s for cosine similarity computing')
        s = time.time()
        print('generating similarities graph')
        sources, targets = cosine_sim.nonzero()
        g = Graph(list(zip(sources.tolist(), targets.tolist())))
        print (str(time.time()-s)+'s for graph generation')
        s= time.time()
        print('creating graph communities')
        clusters= g.community_multilevel(weights = np.exp(k1*cosine_sim.data))
        cluster_labels = clusters.membership
        Z = None
    if clustering_algorithm == 'agglomerative':        
        data = np.array([i[0][0] for i in data_['word_vector']])
        cv_matrix = csr_matrix(np.array(data))
        print('calculating similarity matrix')
        s = time.time()
        #cosine_sim = pairwise_cosine_sparse_sim(cv_matrix, topn = cv_matrix.shape[0], min_value=min_value_,expected_density = expected_density, sparse= False)
        print (str(time.time()-s)+'s for cosine similarity computing')
        
        Z = fastcluster.linkage_vector(np.array(data))
        cluster_labels = scipy.cluster.hierarchy.fcluster(Z, 1-min_value_, criterion='distance', depth=2, R=None, monocrit=None)
        
    
    data_=data_.assign(product_id = cluster_labels)
    
    list_of_labels = list(set(data_['product_id']))
    product_word_vector={}
    for i in list_of_labels:
        by_column_dic_i = data_[data_.product_id == i]
        word_vectors = [vector[0] for vector in by_column_dic_i['word_vector']]
        if len(word_vectors) == 1:   
            product_word_vector[i] = word_vectors
        else:
            product_word_vector[i] = [np.average(np.array(word_vectors),axis=0)]
    
    data_= data_.assign(product_word_vector = 0)
    for i in product_word_vector.keys():
        data_[data_.product_id == i] = data_[data_.product_id == i].assign(product_word_vector =  len(data_[data_.product_id == i])*product_word_vector[i])

    data_ = data_[['ad_title','ad_title_corpus','ad_id','product_id','word_vector','product_word_vector']]
    print (str(time.time()-s)+'s for clusters computation')
    
    if clustering_algorithm == 'community':
        return {'clustered_data':data_ , 'sim_matrix_density': cosine_sim.size/(cosine_sim.shape[0]*cosine_sim.shape[1])}
    if clustering_algorithm == 'agglomerative':
        return {'clustered_data':data_ , 'linkage_matrix': Z}
    


def pairwise_cosine_sparse_sim(csr, topn=4000, min_value = 0.4,expected_density = 1, sparse = True,normalize = True):
    """Computes the cosine distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """   
    if sparse:
        csr = csr_matrix(csr).astype(bool,copy=False).astype(int,copy=False)
        csr = csr.astype('float64',copy=False)
        if normalize:
            csr = normalize(csr, norm='l2', axis=1)
        else:
            pass
        intrsct = dot_product(csr,csr.T, topn, min_value, expected_density)
        #intrsct.data[intrsct.data>=1] = 1        
    else:
        csr = csr.astype('float64',copy=False)
        if normalize:
            csr = normalize(csr, norm='l2', axis=1)
        else:
            pass
        intrsct = dot_product(csr,csr.T, topn, min_value, expected_density)
        intrsct.data[intrsct.data<=0] = 0
    return intrsct        
