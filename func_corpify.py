import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unidecode
import string
import time
import re

def corpify(data,stemming = False):    
    if stemming:
        titulos = data['ad_title']
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
        data = data.assign(ad_title_corpus = corpus)
        data = data.dropna()
    else:
        titulos = data['ad_title']
        new_stopwords = ['unidade','acompanha','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','todo','tudo','quase','caixa','unidade','acompanha','kit','pronto','pronta','entrega', 'frete', 'grátis','garantia','lancamento','lançamento','todo','tudo','quase','caixa','gb','mb','mp','fiscal','nf','ano','completa','box','descric','leia','descrição','descricao','par','barato','importado','combo','unidades','vezes','promoçao','promoção','promocão','promocao','promoçao','novo','brinde','novidade','novidad','oferta','imperdivel','imperdível','luxo','nova','queima','estoque','original','envio','imediato','off','cxs','mais','barata','barato','demais','demai','liquida','pra','oferta','unidad','promoc','promo','tamanho','litro']
        stopwords_ = list(set(stopwords.words('portuguese')+new_stopwords))
        corpus = []
        corpus_=[]
        for i in titulos:    
            titulos_ = unidecode.unidecode(i)
            titulos_ = re.sub(r'[^a-zA-Z]',' ',titulos_)
            titulos_ = titulos_.lower().split()
            titulos_ = [i for i in titulos_ if ((len(i)>2) and ((bool(set('aeiou').intersection(i))) and (bool(set('bcdfghjklmnpqrstvwxyz').intersection(i)))))]
            titulos_ = [word for word in titulos_ if not word in stopwords_]
            corpus.append(titulos_)
            corpus_.append(' '.join(titulos_))
        data = data.assign(ad_title_corpus = corpus_)
        data = data.dropna()
        
        
    return data