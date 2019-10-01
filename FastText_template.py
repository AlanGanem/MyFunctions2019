# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:08:14 2019

@author: PC10
"""
from sklearn.preprocessing import normalize
import gensim, logging
from gensim.models import FastText 
import string, re
from nltk.corpus import stopwords
import unidecode
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
import pickle
import os
import pandas as pd
import func_get_data
import tqdm
from gensim.models import Phrases

titles = pd.read_csv(r'C:\ProductClustering\productsDB\titles_19_03_2019.csv')
titles= titles[['ad_title']]
# from DB
titles = func_get_data.get_data(free_query = "select ad_title from ads_scd", final_date_string = '2001-01-01')
titles = titles.dropna()
titles = titles.assign(ad_title =  titles.ad_title.astype(str))

sample_size = False
if sample_size:
    titulos = pd.DataFrame(titles.sample(sample_size)['ad_title'])
else:
    titulos = titles

trans_table=str.maketrans(string.punctuation+'|',len(string.punctuation+'|')*' ')
new_stopwords = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','c','x','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','é','todo','tudo','quase','caixa','gb','mb','mp','cm','m','mm','p','g','un','unidade','acompanha','kg','mg','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','lançamento','é','todo','tudo','quase','caixa','gb','mb','mp','fiscal','nf','ano','completa','box','descric','leia','descrição','descricao','par','barato','2017','2018','2019','2020','importado','combo','unidades','vezes','promoçao','promoção','promocão','promocao','promoçao','novo','brinde','ft','novidade','novidad','oferta','imperdivel','imperdível','luxo','nova','queima','estoque','original','envio','imediato','off','cxs','mais','barata','barato','demais','demai','liquida','br','pra','ml','oferta','unidad','promoc','promo','tamanho','lt','litro','unico', 'unica']
stopwords_ = list(set(stopwords.words('portuguese')+new_stopwords))
corpus = []
corpus_=[]
for i in tqdm.tqdm(titulos.ad_title):    
    titulos_ = unidecode.unidecode(i)
    titulos_ = re.sub(r'[^a-zA-Z0-9]',' ',titulos_)
    titulos_ = titulos_.lower().split()
    titulos_ = [word for word in titulos_ if not word in stopwords_]
    corpus.append(titulos_)
    corpus_.append(' '.join(titulos_))

titulos = titulos.assign(ad_title_corpus = corpus_)
empty_index = [i for i in range(len(corpus)) if len(corpus[i])==0]
corpus = [corpus[i] for i in range(len(corpus)) if len(corpus[i])>0]
titulos = titulos.drop(empty_index)






#word2vec model
#bigram_transformer = Phrases(corpus)
model = gensim.models.Word2Vec(min_count=5,window = 3,hs=1,size = 50,sg = 1,sample = 0.00001,negative = 0, sorted_vocab = 1)
model.build_vocab(bigram_transformer[corpus])
model.train(corpus, epochs =20, total_examples = model.corpus_count)

f = open(os.path.join(os.path.dirname(r'C:\ProductClustering\ProductsDB\\'), 'model_word2vec_sg_50'), 'wb')
pickle.dump(model,f)
#f = open(os.path.join(os.path.dirname(r'C:\ProductClustering\ProductsDB\\'), 'palavra_composta_transformer'), 'wb')
#pickle.dump(bigram_transformer,f)

palavra = ['banheiro','camping','quimico']
abs(model.wv[palavra]).mean()
model.most_similar(bigram_transformer[palavra])

# train FastText model
#model = gensim.models.Word2Vec(corpus, min_count=2,hs=0)
model_fast_text = FastText(min_count =3,window = 4,sg=0,size = 40 ,sample = 0.0001,negative = 0, sorted_vocab = 1)
model_fast_text.build_vocab(corpus)
model_fast_text.train(corpus, epochs =10, total_examples = model_fast_text.corpus_count)

f = open(os.path.join(os.path.dirname(r'C:\ProductClustering\ProductsDB\\'), 'model_fast_text_sg_80'), 'wb')
pickle.dump(model_fast_text,f)

model_fast_text.wv.most_similar('torneira')
model_fast_text_40.wv.most_similar('torneira')
palavra = 'monocomando'
print(np.linalg.norm(model_fast_text.wv[palavra]))
print(np.linalg.norm(model_fast_text_40.wv[palavra]))
############################################################################

g = os.path.join(os.path.dirname(r'C:\ProductClustering\ProductsDB\\'), 'model_fast_text_sg_40')
model_fast_text_40 = pickle.load(open(g, 'rb'))


rangelen = range(len(corpus))        
data = [np.average(model_fast_text.wv[corpus[i]],axis=0) for i in rangelen]
data = np.array(data)

