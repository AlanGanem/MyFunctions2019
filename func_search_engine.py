from func_corpify import corpify
from sklearn.metrics import pairwise
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unidecode
import string
import numpy as np
from scipy.sparse  import csr_matrix
from sklearn.preprocessing import normalize
from sparse_dot_topn import awesome_cossim_topn as dot_product
import time
import os
import pickle
import re
from func_apply_word_embedings import apply_word_embedings

def search_engine(title, reference_cv_matrix ,vocabulary , metric = 'jaccard' ):   
        
    cv_matrix = reference_cv_matrix
    trans_table=str.maketrans(string.punctuation+'0123456789',len(string.punctuation+'0123456789')*' ')
    new_stopwords = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','c','x','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','é','todo','tudo','quase','caixa','gb','mb','mp','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','lançamento','é','todo','tudo','quase','caixa','gb','mb','mp','fiscal','nf','ano','completa','box','descric','leia','descrição','descricao','par','barato','2017','2018','2019','2020','importado','combo','unidades','vezes','promoçao','promoção','promocão','promocao','promoçao','novo','brinde','ft','novidade','novidad','oferta','imperdivel','imperdível','luxo','nova','queima','estoque','original','envio','imediato','off','cxs','mais','barata','barato','demais','demai','liquida','br','pra','ml','oferta','unidad','promoc','promo','tamanho','lt','litro','metro','metragem','metros']
    texto = title.lower().translate(trans_table).split()    
    ps = SnowballStemmer('portuguese')
    stopwords_ = list(set(stopwords.words('portuguese')+new_stopwords))
    texto_ = [unidecode.unidecode(ps.stem(word)) for word in texto if not word in stopwords_]
    texto  = ' '.join(texto_)
    texto = [texto]
    count_vectorizer = CountVectorizer(binary = True,vocabulary = vocabulary,lowercase = False)
    cv_texto = count_vectorizer.fit_transform(np.array(texto))
    #words_in_voc =  [word for word, index in vocabulary.items() if index in cv_texto.indices]
    range_ = range(len(texto_))
    #words_not_in_voc = [texto_[i] for i in range_ if texto_[i] not in words_in_voc]
    if cv_texto.size > 0:
        if metric == 'cosine':
            similarity = pairwise.cosine_similarity(cv_texto ,cv_matrix)
        if metric =='jaccard':
            similarity = pairwise_jaccard_similarity_sparse(cv_texto ,cv_matrix).A
    else:
        print('no matching result')
        return
    top_3_idx = np.argsort(-similarity[0])[:3]
    top_3_values = [[i,similarity[0,i]] for i in top_3_idx]
    
    #print('words not in vocabulary: '+str(words_not_in_voc))
    return top_3_values

def search_engine_matrix(data_ , productsDB, metric ='cosine'):
    reference_cv_matrix = productsDB['cv_matrix']
    reference_vocabulary = productsDB['vocabulary']
    reference_title_ad_product = productsDB['title_ad_product']
    data_ =  data_.assign(product_id = 0)
    titulos = data_['ad_title']
    s = time.time()
    trans_table=str.maketrans(string.punctuation+'0123456789',len(string.punctuation+'0123456789')*' ')
    new_stopwords = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','c','x','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','é','todo','tudo','quase','caixa','gb','mb','mp','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','lançamento','é','todo','tudo','quase','caixa','gb','mb','mp','fiscal','nf','ano','completa','box','descric','leia','descrição','descricao','par','barato','2017','2018','2019','2020','importado','combo','unidades','vezes','promoçao','promoção','promocão','promocao','promoçao','novo','brinde','ft','novidade','novidad','oferta','imperdivel','imperdível','luxo','nova','queima','estoque','original','envio','imediato','off','cxs','mais','barata','barato','demais','demai','liquida','br','pra','ml','oferta','unidad','promoc','promo','tamanho','lt','litro']
    corpus = []
    stopwords_ = list(set(stopwords.words('portuguese')+new_stopwords))
    ps = SnowballStemmer('portuguese')
    for i in titulos:    
        titulos_ = i.lower().translate(trans_table).split()
        titulos_ = [unidecode.unidecode(ps.stem(word)) for word in titulos_ if not word in stopwords_]
        titulos_ = ' '.join(titulos_)
        corpus.append(titulos_)
    print (str(time.time()-s)+'s for text preprocessing')
    data = data_.copy().assign(ad_title_corpus = corpus)
    count_vectorizer = CountVectorizer(binary = True )
    cv_matrix = count_vectorizer.fit_transform(np.array(data.ad_title_corpus))
    vocabulary = count_vectorizer.vocabulary_   
    sim_matrix = pairwise_cosine_sparse_sim(cv_matrix,reference_cv_matrix, topn = 4000, min_value=0.4)
    iterable = sim_matrix.shape[0]
    labels = []
    for ad in range(iterable):        
        labels.append(reference_title_ad_product.iloc[np.argmax(cv_matrix[ad].A),-1])
    data_=  data_.assign(product_id = labels) 
    
    return  data_

def search_engine_fasttext(data_ , productsDB ,min_sim=0.9,topn = 1,model_name='model_fast_text_sg_40',column_name_db = 'word_vector',column_name_data = 'word_vector',model_folder_path = r'C:\ProductClustering\ProductsDB\fasttext_models\\', metric ='cosine',pre_computed_word_vectors = True):
    
    g = os.path.join(os.path.dirname(model_folder_path), model_name)
    model_fast_text = pickle.load(open(g, 'rb'))
    print('generating reference csr_matrix from data')
    reference_cv_matrix = csr_matrix(np.array([i[0][0] for i in productsDB[column_name_db]]))
    data_ =  data_.assign(product_id_fasttext = 0)
    reference_product_id = productsDB['product_id']

    if not pre_computed_word_vectors:        
        print('applying word embedings to data')                  
        data = apply_word_embedings(data_)
        cv_matrix = csr_matrix(np.array([i[0][0] for i in data[column_name_data]]))   
        print('computing similarity matrix. metric = {}'.format(metric))
        sim_matrix = pairwise_cosine_sparse_sim(cv_matrix,reference_cv_matrix, topn = topn, min_value=min_sim)
        iterable = sim_matrix.shape[0]
        labels = []
        if topn > 1:
            for ad in range(iterable):        
                if np.max(sim_matrix[ad].A) >= min_sim:
                    labels.append(reference_product_id.iloc[np.argmax(sim_matrix[ad].A)])
                else:
                    labels.append(-1)
                print(ad)
        else:
            for ad in range(iterable):        
                if np.max(sim_matrix[ad].A) >= min_sim:
                    labels.append(reference_product_id.iloc[sim_matrix[ad].nonzero()[1][0]])  
                else:
                    labels.append(-1)
                print(ad)
                    
        data_ = data.assign(product_id_fasttext = labels)
    else:
        cv_matrix = csr_matrix(np.array([i[0][0] for i in data_[column_name_data]]))
        print('computing similarity matrix. metric = {}'.format(metric))
        sim_matrix = pairwise_cosine_sparse_sim(cv_matrix,reference_cv_matrix, topn = topn, min_value=min_sim)
        iterable = sim_matrix.shape[0]
        labels = []
        for ad in range(iterable):        
            if np.max(sim_matrix[ad].A) > min_sim:
                labels.append(reference_product_id.iloc[np.argmax(sim_matrix[ad].A)])
            else:
                labels.append(-1)  
            print(ad)
        data_ = data_.assign(product_id_fasttext = labels)
    
    return data_


def pairwise_jaccard_similarity_sparse(vector , csr, epsilon=1):
    """Computes the Jaccard distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """
    
    
    csr = csr_matrix(csr).astype(bool).astype(int)
    vector = vector.astype(bool).astype(int)
    csr_rownnz = csr.getnnz(axis=1)
    vector_rownnz = vector.getnnz(axis=1)
    intrsct = vector*(csr.T)
            
    nnz_i = np.repeat(vector_rownnz, intrsct.getnnz(axis=1))
    unions =nnz_i +  csr_rownnz[intrsct.indices] - intrsct.data
    sims = intrsct.data / unions

    mask = (sims >= 0) & (sims <= epsilon)
    data = sims[mask]
    indices = intrsct.indices[mask]

    rownnz = np.add.reduceat(mask, intrsct.indptr[:-1])
    indptr = np.r_[0, np.cumsum(rownnz)]

    out = csr_matrix((data, indices, indptr), intrsct.shape)        
    return out

def pairwise_cosine_sparse_sim(csr1,csr2, topn=4000, min_value = 0.4, word_embedings = True):
    """Computes the cosine distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """   
    if not word_embedings:
        csr1 = csr_matrix(csr1).astype(bool,copy=False).astype(int,copy=False)
        csr1 = csr1.astype('float64',copy=False)
        csr1 = normalize(csr1, norm='l2', axis=1)
        csr2 = csr_matrix(csr2).astype(bool,copy=False).astype(int,copy=False)
        csr2 = csr2.astype('float64',copy=False)
        csr2 = normalize(csr2, norm='l2', axis=1)
        intrsct = dot_product(csr1,csr2.T, topn, min_value)
        intrsct.data[intrsct.data>=1] = 1
    else:
        csr1 = csr_matrix(csr1).astype('float64',copy=False)
        csr1 = normalize(csr1, norm='l2', axis=1)
        csr2 = csr_matrix(csr2).astype('float64',copy=False)
        csr2 = normalize(csr2, norm='l2', axis=1)
        intrsct = dot_product(csr1,csr2.T, topn, min_value)        
    return intrsct
