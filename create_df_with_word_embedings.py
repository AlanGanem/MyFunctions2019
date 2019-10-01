import pickle
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
def apply_word_embedings(titles,column_name = 'ad_title',model_folder_path = r'C:\ProductClustering\ProductsDB\\' ,model_name = 'model_fast_text_20',)

    g = os.path.join(os.path.dirname(model_folder_path), model_name)
    model_fast_text = pickle.load(open(g, 'rb'))
    
    data = []
    for title in titles.ad_title_corpus:
        data.append(list(normalize(np.sum(model_fast_text.wv[re.sub(r'[^a-zA-Z]',' ',title).lower().split()],axis = 0).reshape(1,-1))))
    titles = titles[['ad_id','ad_title','ad_title_corpus']].assign(word_vector = data)

    return titles
