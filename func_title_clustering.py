from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unidecode
import string
from sklearn.cluster import DBSCAN
import numpy as np
import time
from collections import Counter
from scipy.sparse import csr_matrix



def df_title_clustering(data_, min_size = 2, max_df_ = 1.01, min_df_ = 0, tagging = True, nwords = 5, cumulative = 0.8, sim1 = 0.6,sim2 = 0.5,metric1 = 'cosine', metric2 = 'jaccard'):
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
    
    
    
    s = time.time()
    categories = set(data['category_id'])
    data = data.assign(cluster_label = 0)
    k = 0
    l = 0
    for category in categories :
        ### new sub categories for each category
        titulos = data[data.category_id == category]['ad_title_corpus']
        if len(titulos)>1:            
            documents = np.array(titulos)
            #tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode')
            #cv_matrix = tfidf_vectorizer.fit_transform(documents)
            count_vectorizer = CountVectorizer(binary = True ,max_df = max_df_, min_df = min_df_)
            cv_matrix = count_vectorizer.fit_transform(documents)
            if metric1 == 'cosine':
                clustering = DBSCAN(eps=1-sim1, min_samples=min_size, metric = 'cosine').fit(cv_matrix)
            if metric1 == 'jaccard':
                jac_distance = pairwise_jaccard_sparse(cv_matrix, 1)
                clustering = DBSCAN(eps=1-sim2, min_samples= 1, metric = 'precomputed').fit(jac_distance)                        
            data[data.category_id == category] = data[data.category_id == category].assign(cluster_label = clustering.labels_)
            data[data.category_id == category] = data[data.category_id == category].assign(category_id =str(category)+ '_' +data[data.category_id == category]['cluster_label'].astype(str))
            if len(set(clustering.labels_)) > 1:
                k+=len(set(clustering.labels_))
                l+=Counter(clustering.labels_)[-1]
            else:
                l+=len(clustering.labels_)
        else:
            data[data.category_id == category] = data[data.category_id == category].assign(category_id =str(category)+'_'+str(-1))
            l+=1
    
    del cv_matrix
    del clustering
    
    print(time.time()-s)
    print(str(k)+' new categories found')
    print(str(l)+' itens  uncategorized')
    if tagging:
        print('creating categories blueprint')
        s = time.time()
        data = data.assign(blueprint = -1)
        range_ = set(data.category_id)
        for cat in range_:
            category = data[data.category_id == cat]
            words = []
            range_ = range(len(category))
            for i in range_:
                words += category.ad_title_corpus.iloc[i].split()
            #item_index = list(data[data.category_id == cat].index)
            #words = [corpus[i] for i in item_index]
            c = len(words)
            dic =Counter(words)
            for key in dic:
                dic[key] = dic[key]/c
            dic = dic.most_common()
            del words            
            blueprint = []
            agregate = 0
            i = 0
            while agregate < cumulative:
                agregate+=dic[i][1]
                i+=1
                if agregate == 1:
                    break        
            for k in range(i):
                blueprint.append(dic[k][0])
            if i > nwords:
                blueprint = blueprint[0:nwords]
            del dic
            cat_blueprint = ''
            for i in range(len(blueprint)):
                cat_blueprint += blueprint[i]+' '
            del blueprint
            cat_blueprint = cat_blueprint[:-1]
            
            data[data.category_id == cat] = data[data.category_id == cat].assign(blueprint = cat_blueprint)
            del cat_blueprint
        print(time.time()-s)
        print('creating product clusters')
        s = time.time()
        cats = data[~data.category_id.str.contains('_-1')].groupby('category_id')[['category_id','blueprint']].max()
        blueprints = cats['blueprint']
        ### new sub categories for each category
    
        documents = np.array(blueprints)
        #tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode')
        #cv_matrix = tfidf_vectorizer.fit_transform(documents)
        ### inter categories comparison by tag
        count_vectorizer = CountVectorizer(binary = True,lowercase = False)
        cv_matrix = count_vectorizer.fit_transform(documents)                
        #distance =pairwise_distances(cv_matrix, metric = 'cosine')        
        if metric2 == 'cosine':
            clustering = DBSCAN(eps=1-sim2, min_samples= 1, metric = 'cosine').fit(cv_matrix)
        if metric2 == 'jaccard':
            jac_distance = pairwise_jaccard_sparse(cv_matrix, 1)
            clustering = DBSCAN(eps=1-sim2, min_samples= 1, metric = 'precomputed').fit(jac_distance)            
        #assigning products labels
        cats = cats.assign(cluster_label = clustering.labels_)
        cats_ = set(data.category_id)        
        for category in cats_:
            data[data.category_id == category] = data[data.category_id == category].assign(cluster_label = cats.cluster_label[cats.category_id == category].max())
        data[data.category_id.str.contains('_-1')] = data[data.category_id.str.contains('_-1')].assign(cluster_label = -1)        
        data = data.sort_values(by='cluster_label')
        uncategorized = data[data.cluster_label == -1]
        categorized = data[data.cluster_label != -1]        
        ##### vocabulary and cv_matrix for search engine        
        count_vectorizer = CountVectorizer(binary = True,lowercase = False, vocabulary = vocabulary)
        blueprints_categorized = categorized.groupby('cluster_label')['blueprint'].max()
        documents_categorized = np.array(blueprints_categorized)
        cv_matrix_categorized = count_vectorizer.fit_transform(documents_categorized)        
        print(time.time()-s)
        print(str(len(set(clustering.labels_)))+' products found')        
    return {'categorized':categorized,'uncategorized':uncategorized, 'cv_matrix_categorized':cv_matrix_categorized,'vocabulary': vocabulary}


def pairwise_jaccard_sparse(csr, epsilon):
    """Computes the Jaccard distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """
    assert(0 <= epsilon  <= 1)
    csr = csr_matrix(csr).astype(bool).astype(int)

    csr_rownnz = csr.getnnz(axis=1)
    intrsct = csr.dot(csr.T)

    nnz_i = np.repeat(csr_rownnz, intrsct.getnnz(axis=1))
    unions = nnz_i + csr_rownnz[intrsct.indices] - intrsct.data
    dists = 1.0 - intrsct.data / unions

    mask = (dists >= 0) & (dists <= epsilon)
    data = dists[mask]
    indices = intrsct.indices[mask]

    rownnz = np.add.reduceat(mask, intrsct.indptr[:-1])
    indptr = np.r_[0, np.cumsum(rownnz)]

    out = csr_matrix((data, indices, indptr), intrsct.shape)    
    return out
